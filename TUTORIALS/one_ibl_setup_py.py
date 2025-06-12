import sys
import binascii
import warnings
from pathlib import Path
try:
    import one
    from one.api import ONE
    from one.webclient import AlyxClient
    from one import params
except (ImportError, ModuleNotFoundError) as ex:
    print(sys.version)
    print(f'Script running in environment {sys.prefix}')
    raise ex


def main():
    # OpenAlyx
    oa_par = params.default()
    oa_client_key = params._key_from_url(oa_par.ALYX_URL)
    params.iopar.write(f'{params._PAR_ID_STR}/{oa_client_key}', oa_par)

    # IBL Alyx
    par = params.iopar.from_dict({
        'ALYX_URL': 'https://alyx.internationalbrainlab.org',
        'ALYX_LOGIN': input('Please type your Alyx username: ').strip(),
        'HTTP_DATA_SERVER': 'https://ibl.flatironinstitute.org',
        'HTTP_DATA_SERVER_LOGIN': 'iblmember',
        'HTTP_DATA_SERVER_PWD': binascii.unhexlify(b'477261794d61747465723139').decode('utf-8')
    })

    client_key = params._key_from_url(par.ALYX_URL)
    cache_dir = Path(params.CACHE_DIR_DEFAULT, client_key)
    params.iopar.write(f'{params._PAR_ID_STR}/{client_key}', par)  # Client params

    cache_map = params.iopar.read(f'{params._PAR_ID_STR}/{params._CLIENT_ID_STR}', {'CLIENT_MAP': dict()})
    Path(cache_dir).mkdir(exist_ok=True, parents=True)
    cache_map.CLIENT_MAP[client_key] = str(cache_dir)
    cache_map = cache_map.set('DEFAULT', client_key)

    # Fix other conflicting caches
    client_map = getattr(cache_map, 'CLIENT_MAP', None)
    for k, v in client_map.items():
        if k == client_key:
            continue
        elif v == str(cache_dir):
            new_location = str(Path(params.CACHE_DIR_DEFAULT, k))
            warnings.warn(f'New location for {k}: {new_location}')
            client_map[k] = new_location
    params.iopar.write(f'{params._PAR_ID_STR}/{params._CLIENT_ID_STR}', cache_map)

    ONE.cache_clear()  # in case this script run within same python process
    alyx = AlyxClient()
    alyx.clear_rest_cache()
    alyx.authenticate(username=par.ALYX_LOGIN, force=True)
    print(f'Data will be downloaded to the following location: {alyx.cache_dir}')

if __name__ == '__main__':
    print(sys.version)
    print(f'Script running in environment {sys.prefix}; ONE version {one.__version__}')
    main()
