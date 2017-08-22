mlp_config = {
    'layers': [2, 6, 10],
    'neurons': [16, 1024, 4096],
    'batch_size': [32, 1024, 4096],
    'epochs': 2,
    'optimizer': 'adam',
    'test_times': 5,
    'context': 'multi-gpu',
    'gpus': 4
}

cnn_config = {
    'nets': [
        {
            'kernal_v': 5,
            'kernal_h': 5,
            'stride_v': 1,
            'stride_h': 1,
            'depths': [20, 50],
            'fc_neurons': [500, 10],
            'act': 'tanh',
            'pool': {
                'type': 'max',
                'kernal_v': 2,
                'kernal_h': 2,
                'stride_v': 2,
                'stride_h': 2
            },
            'alias': 'lenet',
            'description': 'lenet'
        }
    ],
    'batch_size': [32, 1024, 4096],
    'epochs': 10,
    'optimizer': 'adam',
    'test_times': 5,
    'context': 'multi-gpu',
    'gpus': 4
}