import hashlib


def normalize_heyex_id(heyex_id: str | int) -> str:
    """
    Normalize the to be a 10 digit number with prefix '0' if necessary, e.g. '123456' -> '0000123456'.
    """
    return f"{int(heyex_id):010d}"

def hashit(s: str) -> str:
    """
    Hash a string using SHA-256.
    """
    return hashlib.sha256(s.encode()).hexdigest()



if __name__ == '__main__':
    print(normalize_heyex_id(123456))
    print(normalize_heyex_id('123456'))
    print(normalize_heyex_id('00123456'))

    print(type(normalize_heyex_id(123456)))