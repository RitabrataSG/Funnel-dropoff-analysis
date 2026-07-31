import pandas as pd

df = pd.read_csv('funnel_events_sample.csv', parse_dates=['timestamp'])
print("Total event rows:", len(df))
print("Unique users:", df['user_id'].nunique())

before = len(df)
df = df.sort_values('timestamp').drop_duplicates(subset=['user_id', 'step'], keep='first')
print(f"Removed {before - len(df)} duplicate step-events")

funnel_order = ['visited_site', 'signup_started', 'details_filled',
                'email_verified', 'purchase_completed']

user_steps = df.groupby('user_id')['step'].apply(set)
skipped_step_users = 0
for s in user_steps:
    idxs = sorted(funnel_order.index(x) for x in s)
    if idxs != list(range(idxs[0], idxs[-1] + 1)):
        skipped_step_users += 1
print(f"Users with a gap in their logged step sequence: {skipped_step_users}")

stage_counts = df.groupby('step')['user_id'].nunique().reindex(funnel_order)

funnel = pd.DataFrame({'stage': funnel_order, 'users': stage_counts.values})
funnel['conversion_from_prev_%'] = (funnel['users'] / funnel['users'].shift(1) * 100).round(1)
funnel.loc[0, 'conversion_from_prev_%'] = 100.0
funnel['conversion_from_start_%'] = (funnel['users'] / funnel['users'].iloc[0] * 100).round(1)
funnel['dropoff_users'] = funnel['users'].shift(1) - funnel['users']
funnel.loc[0, 'dropoff_users'] = 0
funnel['dropoff_%'] = (100 - funnel['conversion_from_prev_%']).round(1)
funnel.loc[0, 'dropoff_%'] = 0.0

print("\n=== FUNNEL SUMMARY ===")
print(funnel.to_string(index=False))

worst_idx_pct = funnel['dropoff_%'][1:].idxmax()
worst_idx_abs = funnel['dropoff_users'][1:].idxmax()

def describe(idx, label):
    f, t = funnel.loc[idx - 1, 'stage'], funnel.loc[idx, 'stage']
    u, p = int(funnel.loc[idx, 'dropoff_users']), funnel.loc[idx, 'dropoff_%']
    print(f"{label}: '{f}' -> '{t}'  ({u} users lost, {p}% drop-off)")

print()
describe(worst_idx_pct, " BIGGEST DROP-OFF (by conversion-rate %)")
describe(worst_idx_abs, "   (for reference) Biggest by raw user count ")

first_ts = df.pivot_table(index='user_id', columns='step', values='timestamp', aggfunc='first')
first_ts = first_ts.reindex(columns=funnel_order)

time_rows = []
for i in range(1, len(funnel_order)):
    a, b = funnel_order[i-1], funnel_order[i]
    pair = first_ts[[a, b]].dropna()
    mins = (pair[b] - pair[a]).dt.total_seconds() / 60
    time_rows.append({'transition': f'{a} -> {b}', 'avg_minutes': round(mins.mean(), 1),
                       'users_with_both_events': len(pair)})
time_df = pd.DataFrame(time_rows)
print("\n=== AVG TIME-TO-CONVERT ===")
print(time_df.to_string(index=False))

def num(u): return int(u.replace('U', ''))
seg_map = df[['user_id']].drop_duplicates()
seg_map['n'] = seg_map['user_id'].apply(num)
maxn = seg_map['n'].max()
seg_map['segment'] = pd.cut(seg_map['n'], bins=[0, maxn/3, 2*maxn/3, maxn+1],
                             labels=['A (early IDs)', 'B (mid IDs)', 'C (late IDs)'])
merged = df.merge(seg_map[['user_id', 'segment']], on='user_id')

seg_rows = []
for seg in ['A (early IDs)', 'B (mid IDs)', 'C (late IDs)']:
    sub = merged[merged['segment'] == seg]
    visited = sub[sub['step'] == 'visited_site']['user_id'].nunique()
    purchased = sub[sub['step'] == 'purchase_completed']['user_id'].nunique()
    rate = round(purchased / visited * 100, 1) if visited else 0
    seg_rows.append({'segment': seg, 'visited': visited, 'purchased': purchased,
                      'overall_conversion_%': rate})
seg_df = pd.DataFrame(seg_rows)
print("\n=== SEGMENT COMPARISON ===")
print(seg_df.to_string(index=False))


funnel.to_csv('funnel_summary.csv', index=False)
time_df.to_csv('time_to_convert.csv', index=False)
seg_df.to_csv('segment_comparison.csv', index=False)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(funnel['stage'], funnel['users'], color='#4C72B0')
for bar, pct in zip(bars, funnel['conversion_from_prev_%']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"{int(bar.get_height())}\n({pct}%)", ha='center', fontsize=9)
ax.set_title('Signup / Checkout Funnel — Users per Stage')
ax.set_ylabel('Unique users')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('funnel_chart.png', dpi=150)
print("\nChart saved to funnel_chart.png")
