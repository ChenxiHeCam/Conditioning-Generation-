# Upload code + checkpoints to the new AutoDL instance (westc:32876).
import os, time, stat, paramiko

HOST, PORT, USER, PW = 'connect.westc.seetacloud.com', 32876, 'root', 'X/VmifKTF+Pz'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=20)
sftp = c.open_sftp()

def put(local, remote):
    sz = os.path.getsize(local)
    t0 = time.time()
    sftp.put(local, remote)
    dt = time.time() - t0
    print(f"  {remote}  ({sz/1e6:.1f} MB, {sz/1e6/max(dt,0.01):.1f} MB/s)", flush=True)

def put_dir_py(local_dir, remote_dir):
    for f in sorted(os.listdir(local_dir)):
        if f.endswith('.py') and not f.startswith('_'):
            put(os.path.join(local_dir, f), f"{remote_dir}/{f}")

for d in ['/root/21cm_gen', '/root/21cm_gen/models', '/root/21cm_gen/utils',
          '/root/autodl-tmp/checkpoints']:
    try: sftp.mkdir(d)
    except IOError: pass

print("== layer 1: GitHub repo base ==")
put_dir_py('C:/Users/1/AppData/Local/Temp/cg_repo', '/root/21cm_gen')
print("== layer 2: local repo (newest variants) ==")
put_dir_py('D:/Astro/repo', '/root/21cm_gen')
print("== layer 3: server-patched scripts ==")
put_dir_py('D:/Astro/repo/scripts_final', '/root/21cm_gen')
print("== models / utils ==")
put('D:/Astro/models/vae.py', '/root/21cm_gen/models/vae.py')
put('D:/Astro/utils/power_spectrum.py', '/root/21cm_gen/utils/power_spectrum.py')
for pkg in ['models', 'utils']:
    with sftp.open(f'/root/21cm_gen/{pkg}/__init__.py', 'w') as f:
        f.write('')

print("== checkpoints (1.6 GB) ==")
ck_dir = 'D:/Astro/repo/ckpts_final'
for f in sorted(os.listdir(ck_dir)):
    put(os.path.join(ck_dir, f), f'/root/autodl-tmp/checkpoints/{f}')

_, o, _ = c.exec_command(
    'ls -la /root/21cm_gen | head -40; echo ---; ls -la /root/autodl-tmp/checkpoints')
print(o.read().decode())
print("UPLOAD DONE")
