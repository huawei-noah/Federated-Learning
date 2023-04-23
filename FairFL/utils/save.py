'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import json

def save_acc_loss(json_file, t, acc, loss):
    result = {}
    result['epoch'] = t
    result['accs'] = list(acc)
    result['losses'] = list(loss)
    with open(json_file, 'a') as f:
        f.write(json.dumps(result, sort_keys=True) + '\n')

