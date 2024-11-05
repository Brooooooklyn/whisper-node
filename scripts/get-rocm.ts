import { execSync } from 'node:child_process'
import * as fs from 'node:fs'
import { parseArgs } from 'node:util'

async function getLatestRocm(): Promise<string> {
  const response = await fetch('https://api.github.com/repos/rocm/rocm/releases/latest')
  const data = await response.json()
  const [_, verStr] = data.tag_name.split('-')
  return verStr
}

interface OsReleaseMeta {
  [key: string]: string
}

function getOsReleaseMeta(): OsReleaseMeta | undefined {
  try {
    const osRel = fs.readFileSync('/etc/os-release', 'utf8')
    const kvs: OsReleaseMeta = {}

    for (const line of osRel.split('\n')) {
      if (line.trim()) {
        const [k, v] = line.trim().split('=', 2)
        kvs[k] = v.replace(/"/g, '')
      }
    }

    return kvs
  } catch (err) {
    return undefined
  }
}

interface Args {
  rocmVersion: string
  sudo: boolean
  jobName?: string
  buildNum?: string
}

function parseRocmArgs(): Args {
  const { values } = parseArgs({
    options: {
      'rocm-version': {
        type: 'string',
        default: 'latest',
      },
      'job-name': {
        type: 'string',
      },
      'build-num': {
        type: 'string',
      },
      sudo: {
        type: 'boolean',
        default: false,
      },
    },
    args: process.argv.slice(2),
  })

  return {
    rocmVersion: values['rocm-version'],
    jobName: values['job-name'],
    buildNum: values['build-num'],
    sudo: values.sudo,
  }
}

const args = parseRocmArgs()

class System {
  constructor(
    public readonly pkgbin: string,
    public readonly rocmPackageList: string[],
    public readonly sudo: boolean,
  ) {}

  installPackages(packageSpecs: string[]): void {
    const cmd = withSudo(this.sudo, [this.pkgbin, 'install', '-y', ...packageSpecs])

    const env = { ...process.env }
    if (this.pkgbin === 'apt') {
      env.DEBIAN_FRONTEND = 'noninteractive'
    }

    console.info(`Running ${cmd}`)
    execSync(cmd.join(' '), { env, stdio: 'inherit' })
  }

  async installRocm(): Promise<void> {
    this.installPackages(this.rocmPackageList)
  }
}

const UBUNTU = new System('apt', ['rocm-dev', 'rocm-libs'], args.sudo)

const RHEL8 = new System(
  'dnf',
  [
    'libdrm-amdgpu',
    'rocm-dev',
    'rocm-ml-sdk',
    'miopen-hip',
    'miopen-hip-devel',
    'rocblas',
    'rocblas-devel',
    'rocsolver-devel',
    'rocrand-devel',
    'rocfft-devel',
    'hipfft-devel',
    'hipblas-devel',
    'rocprim-devel',
    'hipcub-devel',
    'rccl-devel',
    'hipsparse-devel',
    'hipsolver-devel',
  ],
  args.sudo,
)

interface Version {
  major: number
  minor: number
  rev?: number
}

function parseVersion(versionStr: string | Version): Version {
  if (typeof versionStr === 'string') {
    const parts = versionStr.split('.')
    const version: Version = {
      major: parseInt(parts[0].trim()),
      minor: parseInt(parts[1].trim()),
      rev: parts.length > 2 ? parseInt(parts[2].trim()) : undefined,
    }
    return version
  }
  return versionStr
}

function getSystem(): System {
  const md = getOsReleaseMeta()
  if (!md) throw new Error('Could not get OS release metadata')

  if (md.ID === 'ubuntu') {
    return UBUNTU
  }

  if (['almalinux', 'rhel', 'fedora', 'centos'].includes(md.ID)) {
    if (md.PLATFORM_ID === 'platform:el8') {
      return RHEL8
    }
  }

  throw new Error(`No system for ${md}`)
}

async function setupInternalRepo(
  system: System,
  sudo: boolean,
  rocmVersion: string,
  jobName: string,
  buildNum: string,
): Promise<void> {
  system.installPackages(['wget'])

  await installAmdgpuInstallerInternal(rocmVersion)

  const response = await fetch(`http://rocm-ci.amd.com/job/${jobName}/${buildNum}/artifact/amdgpu_kernel_info.txt`)
  const amdgpuBuild = (await response.text()).trim()

  const cmd = withSudo(sudo, ['amdgpu-repo', `--amdgpu-build=${amdgpuBuild}`, `--rocm-build=${jobName}/${buildNum}`])

  console.info(`Running ${cmd}`)
  execSync(cmd.join(' '), { stdio: 'inherit' })

  const installCmd = ['amdgpu-install', '--no-dkms', '--usecase=rocm', '-y']

  const env = { ...process.env }
  if (system.pkgbin === 'apt') {
    env.DEBIAN_FRONTEND = 'noninteractive'
  }

  console.info(`Running ${installCmd}`)
  execSync(installCmd.join(' '), { env, stdio: 'inherit' })
}

async function installRocm(
  rocmVersion: string,
  sudo: boolean,
  jobName?: string | null,
  buildNum?: string | null,
): Promise<void> {
  const s = getSystem()

  if (jobName && buildNum) {
    await setupInternalRepo(s, sudo, rocmVersion, jobName, buildNum)
  } else {
    if (s === RHEL8) {
      await setupReposEl8(rocmVersion)
    } else if (s === UBUNTU) {
      await setupReposUbuntu(rocmVersion)
    } else {
      throw new Error('Platform not supported')
    }
  }

  await s.installRocm()
}

async function installAmdgpuInstallerInternal(rocmVersion: string): Promise<void> {
  const md = getOsReleaseMeta()
  if (!md) throw new Error('Could not get OS release metadata')

  const [url, fn] = buildInstallerUrl(rocmVersion, md)

  try {
    console.info(`Downloading from ${url}`)
    const response = await fetch(url)
    const buffer = await response.arrayBuffer()
    await fs.promises.writeFile(fn, Buffer.from(buffer))

    const system = getSystem()
    const cmd = [system.pkgbin, 'install', '-y', `./${fn}`]
    execSync(cmd.join(' '), { stdio: 'inherit' })
  } finally {
    try {
      await fs.promises.unlink(fn)
    } catch (err) {
      if ((err as any).code !== 'ENOENT') throw err
    }
  }
}

function buildInstallerUrl(rocmVersion: string, metadata: OsReleaseMeta): [string, string] {
  const rv = parseVersion(rocmVersion)
  const baseUrl = 'http://artifactory-cdn.amd.com/artifactory/list'

  if (metadata.ID === 'ubuntu') {
    const packageName = `amdgpu-install-internal_${rv.major}.${rv.minor}-${metadata.VERSION_ID}-1_all.deb`
    const url = `${baseUrl}/amdgpu-deb/${packageName}`
    return [url, packageName]
  } else if (metadata.PLATFORM_ID === 'platform:el8') {
    const packageName = `amdgpu-install-internal-${rv.major}.${rv.minor}_8-1.noarch.rpm`
    const url = `${baseUrl}/amdgpu-rpm/rhel/${packageName}`
    return [url, packageName]
  }

  throw new Error(`Platform not supported: ${metadata}`)
}

const APT_RADEON_PIN_CONTENT = `Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
`

async function setupReposUbuntu(rocmVersionStr: string): Promise<void> {
  const rv = parseVersion(rocmVersionStr)
  let versionStr = rocmVersionStr

  if (rv.rev === 0) {
    versionStr = `${rv.major}.${rv.minor}`
  }

  execSync(withSudo(args.sudo, ['apt-get', 'update']).join(' '), { stdio: 'inherit' })

  const s = getSystem()
  s.installPackages(['wget', 'sudo', 'gnupg'])

  const md = getOsReleaseMeta()
  if (!md) throw new Error('Could not get OS release metadata')
  const codename = md.VERSION_CODENAME

  execSync(`wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | ${args.sudo ? 'sudo ' : ''}apt-key add -`, {
    stdio: 'inherit',
  })

  tee(
    args.sudo,
    `deb [arch=amd64] https://repo.radeon.com/amdgpu/${versionStr}/ubuntu ${codename} main\n`,
    '/etc/apt/sources.list.d/amdgpu.list',
  )

  tee(
    args.sudo,
    `deb [arch=amd64] https://repo.radeon.com/rocm/apt/${versionStr} ${codename} main\n`,
    '/etc/apt/sources.list.d/rocm.list',
  )

  tee(args.sudo, APT_RADEON_PIN_CONTENT, '/etc/apt/preferences.d/rocm-pin-600')

  execSync(withSudo(args.sudo, ['apt-get', 'update']).join(' '), { stdio: 'inherit' })
}

async function setupReposEl8(rocmVersionStr: string): Promise<void> {
  tee(
    args.sudo,
    `[ROCm]
name=ROCm
baseurl=http://repo.radeon.com/rocm/rhel8/${rocmVersionStr}/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
`,
    '/etc/yum.repos.d/rocm.repo',
  )

  tee(
    args.sudo,
    `[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/${rocmVersionStr}/rhel/8.8/main/x86_64/
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
`,
    '/etc/yum.repos.d/amdgpu.repo',
  )
}

let rocmVersion: string

if (args.rocmVersion === 'latest') {
  try {
    rocmVersion = await getLatestRocm()
    console.log(`Latest ROCm release: ${rocmVersion}`)
  } catch (err) {
    console.error("Latest ROCm lookup failed. Please use '--rocm-version' to specify a version instead.")
    process.exit(-1)
  }
} else {
  rocmVersion = args.rocmVersion
}

function withSudo(sudo: boolean, cmd: string[]) {
  if (sudo) {
    return ['sudo', ...cmd]
  }
  return cmd
}

function tee(sudo: boolean, content: string, file: string) {
  const contentLines = content.split('\n')
  contentLines.forEach((line) => {
    execSync(`echo "${line}" | ${sudo ? 'sudo ' : ''}tee -a ${file}`, { stdio: 'inherit' })
  })
}

await installRocm(rocmVersion, args.sudo, args.jobName, args.buildNum)
