# JanusForge Milestone 1 - reproducible race labeling pipeline
import pandas as pd

RACES = {9472: 'Bahrain', 9488: 'Australia', 9539: 'Spain', 9550: 'Austria', 9636: 'São Paulo'}


def load_race_table(race_path, name, timestamp):
    df = pd.read_csv(race_path / f'{name}.csv')
    df[timestamp] = pd.to_datetime(df[timestamp], format='mixed', utc=True, errors='coerce')
    return df.dropna(subset=[timestamp])


def build_race_labels(session_key, name, data_dir):
    folder = data_dir / '2024' / 'races' / str(session_key)
    pos = load_race_table(folder, 'position', 'date').sort_values('date')
    laps = load_race_table(folder, 'laps', 'date_start')
    gaps = load_race_table(folder, 'intervals', 'date')
    pits = load_race_table(folder, 'pit', 'date')
    overtakes = load_race_table(folder, 'overtakes', 'date')
    flags = load_race_table(folder, 'race_control', 'date')

    laps['lap_duration'] = pd.to_numeric(laps['lap_duration'], errors='coerce')
    valid_laps = laps[laps['lap_duration'].between(30, 600)]
    start = valid_laps['date_start'].min()
    end = (valid_laps['date_start'] + pd.to_timedelta(valid_laps['lap_duration'], unit='s')).max()
    finish_times = flags.loc[flags['message'].eq('SESSION FINISHED'), 'date']
    finish = finish_times.min() if not finish_times.empty else end
    if pd.isna(finish):
        raise ValueError(f'No race finish for {name}')
    times = pd.date_range(start, end - pd.Timedelta(seconds=60), freq='10s')

    # Reconstruct the current race position at each 10-second snapshot.
    snapshots = []
    for d, history in pos.groupby('driver_number'):
        current = pd.merge_asof(
            pd.DataFrame({'time': times}),
            history[['date', 'position']].rename(columns={'date': 'last_position_update'}).sort_values('last_position_update'),
            left_on='time', right_on='last_position_update', direction='backward'
        )
        current['driver_number'] = d
        snapshots.append(current)
    snapshots = pd.concat(snapshots, ignore_index=True).dropna(subset=['position'])
    snapshots['position'] = snapshots['position'].astype(int)
    ambiguous_times = snapshots.loc[snapshots.duplicated(['time', 'position'], keep=False), 'time'].unique()
    snapshots = snapshots[~snapshots['time'].isin(ambiguous_times)].copy()
    ahead = snapshots[['time', 'position', 'driver_number']].rename(columns={'position': 'ahead_position', 'driver_number': 'driver_ahead'})
    snapshots['ahead_position'] = snapshots['position'] - 1
    pairs = snapshots.merge(ahead, on=['time', 'ahead_position'], how='left', validate='many_to_one')
    pairs = pairs[(pairs['position'] > 1) & pairs['driver_ahead'].notna()].copy()
    pairs['driver_ahead'] = pairs['driver_ahead'].astype(int)

    gaps['gap_seconds'] = pd.to_numeric(gaps['interval'], errors='coerce')
    joined = pd.merge_asof(
        pairs.sort_values('time'),
        gaps[['date', 'driver_number', 'gap_seconds']].rename(columns={'date':'gap_timestamp'}).sort_values('gap_timestamp'),
        left_on='time', right_on='gap_timestamp', by='driver_number', direction='backward'
    )
    gap_age = (joined['time'] - joined['gap_timestamp']).dt.total_seconds()
    joined = joined[
        joined['gap_seconds'].between(0, 3, inclusive='right')
        & gap_age.between(0, 20)
        & (joined['time'] >= start + pd.Timedelta(minutes=2))
        & (joined['time'] + pd.Timedelta(seconds=60) <= end)
    ].copy()

    # Exclude windows around pit stops (either driver).
    pit_ok = pd.Series(True, index=joined.index)
    for pit in pits.itertuples(index=False):
        involved = (joined['driver_number'] == pit.driver_number) | (joined['driver_ahead'] == pit.driver_number)
        nearby = joined['time'].between(pit.date - pd.Timedelta(seconds=90), pit.date + pd.Timedelta(seconds=90))
        pit_ok &= ~(involved & nearby)
    pairs = joined.loc[pit_ok, ['time', 'driver_number', 'driver_ahead', 'position', 'gap_seconds']].copy().reset_index(drop=True)
    pairs['observation_id'] = pairs.index

    # Search for direct overtakes in the NEXT 60 seconds.
    events = overtakes.rename(columns={'date': 'overtake_time', 'overtaking_driver_number': 'driver_number', 'overtaken_driver_number': 'driver_ahead'})[
        ['overtake_time', 'driver_number', 'driver_ahead']].drop_duplicates()
    matched = pairs.merge(events, on=['driver_number', 'driver_ahead'], how='inner')
    matched = matched[
        (matched['overtake_time'] > matched['time'])
        & (matched['overtake_time'] <= matched['time'] + pd.Timedelta(seconds=60))
    ].copy()

    histories = {d: h.set_index('date')['position'].sort_index() for d, h in pos.groupby('driver_number')}
    def position_at(driver, moment):
        h = histories.get(driver)
        if h is None:
            return None
        idx = h.index.searchsorted(moment, side='right') - 1
        return None if idx < 0 else int(h.iloc[idx])

    def valid_pass(e):
        t, a, b = e.overtake_time, e.driver_number, e.driver_ahead
        before_a = position_at(a, t - pd.Timedelta(seconds=5))
        before_b = position_at(b, t - pd.Timedelta(seconds=5))
        after_a = position_at(a, t + pd.Timedelta(seconds=5))
        after_b = position_at(b, t + pd.Timedelta(seconds=5))
        later_a = position_at(a, t + pd.Timedelta(seconds=35))
        later_b = position_at(b, t + pd.Timedelta(seconds=35))
        pit_near = pits['driver_number'].isin([a,b]) & pits['date'].between(t - pd.Timedelta(seconds=120), t + pd.Timedelta(seconds=120))
        return (
            before_a is not None and before_b is not None and before_a == before_b + 1
            and after_a is not None and after_b is not None and after_a < after_b
            and later_a is not None and later_b is not None and later_a < later_b
            and not pit_near.any()
        )
    unique_events = matched[['overtake_time', 'driver_number', 'driver_ahead']].drop_duplicates()
    validation = unique_events.copy()
    validation['valid'] = [valid_pass(e) for e in unique_events.itertuples(index=False)]
    matched = matched.merge(validation, on=['overtake_time', 'driver_number', 'driver_ahead'], validate='many_to_one')
    positives = set(matched.loc[matched['valid'], 'observation_id'])
    unclear = set(matched['observation_id']) - positives

    pairs['label'] = pd.Series(0, index=pairs.index, dtype='Int64')
    pairs.loc[pairs['observation_id'].isin(positives), 'label'] = 1
    pairs.loc[pairs['observation_id'].isin(unclear), 'label'] = pd.NA

    # If either driver participates in a different overtake, don't label as negative.
    neg = pairs[pairs['label'].eq(0)]
    watched = pd.concat([
        neg[['observation_id', 'time', 'driver_number']].rename(columns={'driver_number': 'watched_driver'}),
        neg[['observation_id', 'time', 'driver_ahead']].rename(columns={'driver_ahead': 'watched_driver'})
    ], ignore_index=True)
    event_drivers = pd.concat([
        overtakes[['date', 'overtaking_driver_number']].rename(columns={'date':'event_time','overtaking_driver_number':'watched_driver'}),
        overtakes[['date', 'overtaken_driver_number']].rename(columns={'date':'event_time','overtaken_driver_number':'watched_driver'})
    ], ignore_index=True).drop_duplicates()
    conflict = watched.merge(event_drivers, on='watched_driver', how='inner')
    conflict = conflict[(conflict['event_time'] > conflict['time']) & (conflict['event_time'] <= conflict['time'] + pd.Timedelta(seconds=60))]
    pairs.loc[pairs['observation_id'].isin(conflict['observation_id']) & pairs['label'].eq(0).fillna(False), 'label'] = pd.NA

    # Exclude periods around documented yellow/double-yellow flags and official finish.
    yellow_mask = pd.Series(False, index=pairs.index)
    for ev in flags[flags['flag'].isin(['YELLOW','DOUBLE YELLOW'])].itertuples(index=False):
        sector_match = flags['sector'].eq(ev.sector) if pd.notna(ev.sector) else flags['sector'].isna()
        scope_match = flags['scope'].eq(ev.scope) if pd.notna(ev.scope) else flags['scope'].isna()
        clears = flags.loc[(flags['flag'] == 'CLEAR') & sector_match & scope_match & (flags['date'] > ev.date), 'date']
        clear_time = clears.min() if not clears.empty else finish
        yellow_mask |= ((pairs['time'] - pd.Timedelta(seconds=30) <= clear_time) & (pairs['time'] + pd.Timedelta(seconds=60) >= ev.date))
    pairs = pairs.loc[~yellow_mask & (pairs['time'] + pd.Timedelta(seconds=60) <= finish)].copy()

    # Require lap coverage for both cars. A one-lap-number difference at the line is valid.
    lap_lookup = laps[['date_start','driver_number','lap_number','lap_duration']].copy()
    lap_lookup['lap_end'] = lap_lookup['date_start'] + pd.to_timedelta(lap_lookup['lap_duration'], unit='s')
    def add_lap(data, driver_col, prefix):
        right = lap_lookup[['date_start','driver_number','lap_number','lap_end']].rename(columns={
            'date_start':prefix+'_start','driver_number':driver_col,'lap_number':prefix+'_lap','lap_end':prefix+'_end'
        })
        out = pd.merge_asof(data.sort_values('time'), right.sort_values(prefix+'_start'), left_on='time', right_on=prefix+'_start', by=driver_col, direction='backward')
        good = out[prefix+'_lap'].notna() & out[prefix+'_end'].notna() & (out['time'] <= out[prefix+'_end'] + pd.Timedelta(seconds=3))
        return out, good
    pairs, good_follower = add_lap(pairs, 'driver_number', 'follower')
    pairs, good_ahead = add_lap(pairs, 'driver_ahead', 'ahead')
    # Series indexes preserved by merge_asof, but calculate active masks again on merged frame.
    pairs = pairs[
        pairs['follower_lap'].notna() & pairs['ahead_lap'].notna()
        & (pairs['time'] <= pairs['follower_end'] + pd.Timedelta(seconds=3))
        & (pairs['time'] <= pairs['ahead_end'] + pd.Timedelta(seconds=3))
        & pairs['label'].notna()
    ].copy()
    pairs['label'] = pairs['label'].astype(int)
    pairs['session_key'] = session_key
    pairs['race'] = name
    result = pairs[['session_key','race','time','driver_number','driver_ahead','position','gap_seconds','follower_lap','ahead_lap','label']].reset_index(drop=True)

    special = flags[flags['message'].fillna('').str.contains(r'\b(?:SAFETY CAR|VSC|RED FLAG)\b', case=False, regex=True)][['date','message','flag']]
    stats = {'race':name,'rows':len(result),'positive':int(result['label'].sum()),'negative':int((result['label']==0).sum()),'validated_events':int(validation['valid'].sum()),'neutralization_messages':len(special)}
    return result, stats, special


processed_dir = DATA_DIR / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)
all_results = []
all_stats = []
all_specials = []
for session_key, name in RACES.items():
    data, stats, special = build_race_labels(session_key, name, DATA_DIR)
    path = processed_dir / f'{session_key}_labeled.csv'
    data.to_csv(path, index=False)
    all_results.append(data)
    all_stats.append(stats)
    if len(special):
        special = special.copy()
        special.insert(0, 'race', name)
        all_specials.append(special)
    print(f"{name}: {stats['rows']:,} rows, {stats['positive']} positives / {stats['negative']} negatives -> {path}")

all_labeled = pd.concat(all_results, ignore_index=True)
all_labeled.to_csv(processed_dir / 'all_races_labeled.csv', index=False)
display(pd.DataFrame(all_stats))
print('Combined dataset:', len(all_labeled), 'rows')
print('Any missing:', bool(all_labeled.isna().any().any()))
# Verify against the manually audited Bahrain output, if present.
manual_bahrain = processed_dir / 'bahrain_labeled.csv'
if manual_bahrain.exists():
    reference = pd.read_csv(manual_bahrain)
    generated = all_results[0]
    keys = ['time', 'driver_number', 'driver_ahead']
    reference['time'] = pd.to_datetime(reference['time'], format='mixed', utc=True)
    observed = generated.merge(reference[keys + ['label']], on=keys, how='outer', suffixes=('_generated', '_manual'), indicator=True)
    wrong = observed[(observed['_merge'] != 'both') | (observed['label_generated'] != observed['label_manual'])]
    print(f'Bahrain consistency check: {len(wrong)} differing rows/labels')
    if len(wrong):
        display(wrong.head(10))
else:
    print('Bahrain reference not found; compare with 3,838 rows / 126 positive manually.')
if all_specials:
    print('WARNING: Review safety-car/VSC/red-flag messages before calling data finalized:')
    display(pd.concat(all_specials, ignore_index=True).head(60))
