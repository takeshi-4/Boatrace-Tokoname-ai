import runpy
import json
from datetime import datetime

# Execute loader module file and get its globals
loader_globals = runpy.run_path('boatrace-master/src/data_preparing/loader.py')

make_df = loader_globals.get('make_race_result_df')
if make_df is None:
    raise RuntimeError('make_race_result_df not found in loader module')

# Run the loader to produce dataframe and populate LAST_RUN_DIAGNOSTICS
print('Running make_race_result_df()...')
df = make_df()
print('Done. Rows:', len(df))

LAST = loader_globals.get('LAST_RUN_DIAGNOSTICS', {})

diagnostics = {
    'generated_at': datetime.now().isoformat(),
    'rows': len(df),
    'weather_nonnull': int(df['weather'].notnull().sum()) if 'weather' in df.columns else 0,
    'columns': list(df.columns),
    'LAST_RUN_DIAGNOSTICS': LAST,
}

with open('diagnostics/tokoname_diagnostics.json', 'w', encoding='utf-8') as f:
    json.dump(diagnostics, f, ensure_ascii=False, indent=2, default=str)

print('Wrote diagnostics/tokoname_diagnostics.json')
