import json
for tag, path in [
    ('v4logk (64³ patch)',  '/root/autodl-tmp/full_field_100_v4logk/metrics_kcut05.json'),
    ('128 ep264 (LDM finetune from 249)', '/root/autodl-tmp/full_field_100_128_ep264/metrics_kcut05.json'),
]:
    d = json.load(open(path))
    s = d['summary']
    print(f"=== {tag} ===")
    for k, v in s.items():
        print(f"  {k:10s} median={v['median']:.4f}  mean={v['mean']:.4f}  IQR=[{v['iqr'][0]:.4f},{v['iqr'][1]:.4f}]")
