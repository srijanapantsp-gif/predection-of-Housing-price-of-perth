try:
    import lightgbm as lgb
    print('LightGBM installed successfully!')
    print(f'Version: {lgb.__version__}')
except ImportError as e:
    print(f'LightGBM not installed. Error: {e}')
except Exception as e:
    print(f'Error importing LightGBM: {e}')





