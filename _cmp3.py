import json
for tag, path in [
    ('v2  (30step, k>0.05)',    '/root/autodl-tmp/full_field_100_s30/metrics_kcut05.json'),
    ('v3lowk (30step, k>0.05)', '/root/autodl-tmp/full_field_100_v3lowk/metrics_kcut05.json'),
    ('v4logk (30step, k>0.05)', '/root/autodl-tmp/full_field_100_v4logk/metrics_kcut05.json'),
]:
    d = json.load(open(path))
    s = d['summary']
    print(f"=== {tag} ===")
    for k, v in s.items():
        print(f"  {k:10s} median={v['median']:.4f}  mean={v['mean']:.4f}  IQR=[{v['iqr'][0]:.4f},{v['iqr'][1]:.4f}]")
