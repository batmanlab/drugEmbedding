#!/usr/bin/env python

# request API lib setup using OpenFDA and query strings from
# https://github.com/amida-tech/rxnorm-js

# pollack api key: 55pQB5eYrVnM2ooFIwr9CYVuFqEwLuwuy5z9Jmw4

import time
import requests
import pandas as pd
from pandas.io.json import json_normalize


def make_request(gen_name=None, rxnorm=None):
    if gen_name is not None and rxnorm is not None:
        raise ValueError('Only specify gen_name or rxnorm, not both')
    if gen_name is not None:
        query_string = (
            'https://api.fda.gov/drug/event.json?api_key=55pQB5eYrVnM2ooFIwr9CYVuFqEwLuwuy5z9Jmw4&'
            f'search=patient.drug.openfda.generic_name:"{gen_name}"&count='
            'patient.reaction.reactionmeddrapt.exact')
    elif rxnorm is not None:
        query_string = (
            'https://api.fda.gov/drug/event.json?api_key=55pQB5eYrVnM2ooFIwr9CYVuFqEwLuwuy5z9Jmw4&'
            f'search=patient.drug.openfda.rxcui:"{rxnorm}"&count='
            'patient.reaction.reactionmeddrapt.exact')

    print(query_string)
    for i in range(6):
        req = requests.get(query_string)
        if req.status_code == 200:
            return req
        elif req.status_code in [500, 502, 504]:
            time.sleep(1+i)
        else:
            print(f'Request for {query_string} failed with {req.status_code}')
            return None
    print(f'Request for {query_string} failed')
    return None


def get_flat_pandas(gen_name=None, rxnorm=None):
    req_name = gen_name if not None else rxnorm
    req = make_request(gen_name, rxnorm)
    if req is None:
        return None
    try:
        df_api = json_normalize(req.json()['results'])
    except KeyError:
        print(f'{req_name} not found!')
        return None
    return df_api.pivot_table(values='count', columns='term',
                              aggfunc='first').rename(index={'count': f'{req_name}'})
