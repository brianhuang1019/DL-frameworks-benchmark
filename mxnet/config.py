mlp_config = {
    'layers': [2, 4, 6, 8, 10],
    'neurons': [16, 32, 128, 512, 1024],
    'batch_size': [32, 64, 128, 512, 1024],
    'epochs': 30,
    'optimizer': 'adam',
    'test_times': 10,
    'context': 'gpu'
}

cnn_config = {
    'nets': [
        {
            'kernal_v': 5,
            'kernal_h': 5,
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
            'description': 'lenet'
        }
    ],
    'batch_size': [32, 64, 128, 512, 1024],
    'epochs': 30,
    'optimizer': 'adam',
    'test_times': 10,
    'context': 'gpu'
}