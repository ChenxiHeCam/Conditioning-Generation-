# One-shot CSD3 session (needs fresh TOTP as argv[1]):
#   1. inspect dataset sizes on CSD3
#   2. install CSD3 pubkey -> AutoDL weste
#   3. launch rsync CSD3 -> AutoDL inside tmux (survives logout)
# Usage: python _csd3_kickoff.py 123456
import sys, time, paramiko

TOTP = sys.argv[1]
CSD3_PW = 'hcxhaoshuaI1/'
ADL_HOST, ADL_PORT, ADL_PW = 'connect.weste.seetacloud.com', 46022, 'x0K6BWzk2rSi'
DATA = '/home/ch2067/rds/hpc-work/ASR21cm/datasets'

def csd3_auth(title, instructions, fields):
    answers = []
    for prompt, echo in fields:
        pl = prompt.lower()
        if 'password' in pl:
            answers.append(CSD3_PW)
        else:  # TOTP / verification
            answers.append(TOTP)
    return answers

t = paramiko.Transport(('login-cpu.hpc.cam.ac.uk', 22))
t.connect()
t.auth_interactive('ch2067', csd3_auth)
assert t.is_authenticated(), 'CSD3 auth failed'
print('CSD3 auth OK')
cl = paramiko.SSHClient(); cl._transport = t

def run(cmd, timeout=120):
    _, o, e = cl.exec_command(cmd, timeout=timeout)
    out = o.read().decode(); err = e.read().decode()
    if out: print(out)
    if err: print('STDERR:', err[:500])
    return out

print('=== dataset inventory ===')
run(f'du -sh {DATA}/*/ 2>/dev/null; echo ---; ls {DATA}/; echo ---; '
    f'for d in {DATA}/*/; do echo $d; ls $d | head -5; done')

print('=== ensure CSD3 keypair, install on AutoDL ===')
pub = run('test -f ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q; cat ~/.ssh/id_ed25519.pub').strip()

a = paramiko.SSHClient(); a.set_missing_host_key_policy(paramiko.AutoAddPolicy())
a.connect(ADL_HOST, port=ADL_PORT, username='root', password=ADL_PW, timeout=20)
_, o, _ = a.exec_command(f'grep -qF "{pub}" ~/.ssh/authorized_keys 2>/dev/null || echo "{pub}" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; echo INSTALLED')
print(o.read().decode())

print('=== test CSD3 -> AutoDL ssh ===')
run(f'ssh -p {ADL_PORT} -o StrictHostKeyChecking=no -o BatchMode=yes root@{ADL_HOST} "echo CSD3_TO_ADL_OK; df -h /root/autodl-tmp | tail -1"')

print('=== launch rsync in tmux ===')
rsync_cmd = (f'rsync -a --partial --info=progress2 '
             f'-e "ssh -p {ADL_PORT} -o StrictHostKeyChecking=no" '
             f'{DATA}/ root@{ADL_HOST}:/root/autodl-tmp/ASR21cm/')
run(f'tmux kill-session -t adl_xfer 2>/dev/null; '
    f'tmux new-session -d -s adl_xfer \'{rsync_cmd} 2>&1 | tee ~/adl_xfer.log\'; '
    f'sleep 5; tmux ls; tail -5 ~/adl_xfer.log 2>/dev/null')
print('=== done — monitor from AutoDL side with: du -sh /root/autodl-tmp/ASR21cm ===')
