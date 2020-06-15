TRAIN_START_DATE = "2011-01-03"
TRAIN_STOP_DATE = "2012-12-31"

VAL_START_DATE = "2013-01-02"
VAL_STOP_DATE = "2013-12-31"

TEST_START_DATE = "2014-01-02"
TEST_STOP_DATE = "2015-04-13"

SPYfeatures = {"MLP":[1,3,5,7,8,9,12],
               "RNN":[1,2,3,5,7,8,9,10,12],
               "PSN":[1,2,3,5,7,8,9,10,12]
              }
DIAfeatures = {"MLP":[2,4,5,7,9,10,11],
               "RNN":[1,3,4,6,7,8,9,10],
               "PSN":[1,2,5,6,8,9,10]
              }
QQQfeatures = {"MLP":[1,2,3,5,6,8,10,11,12],
               "RNN":[1,4,5,6,7,9,10,12],
               "PSN":[2,4,5,6,7,8,9,10,11]
              }

train_parameters = {
    "SPY" : {
        "MLP" : {
            "optim" : "SGD",
            "lr" : 0.003,
            "momentum" : 0.004,
            "epochs" : 30000,
            "weights_init" : "norm",
            "input_size" : 7,
            "hidden_size" : 6,
            "output_size" : 1,
        },
        
        "RNN" : {
            "optim" : "SGD",
            "lr" : 0.003,
            "momentum" : 0.005,
            "epochs" : 40000,
            "weights_init" : "norm",
            "input_size" : 9,
            "hidden_size" : 6,
            "output_size" : 1,
        },
        
        "PSN" : {
            "optim" : "SGD",
            "lr" : 0.4,
            "momentum" : 0.5,
            "epochs" : 40000,
            "weights_init" : "norm",
            "input_size" : 9,
            "hidden_size" : 5,
            "output_size" : 1,
        }
    },
    
    "DIA" : {
        "MLP" : {
            "optim" : "SGD",
            "lr" : 0.002,
            "momentum" : 0.005,
            "epochs" : 45000,
            "weights_init" : "norm",
            "input_size" : 7,
            "hidden_size" : 9,
            "output_size" : 1,
        },
        
        "RNN" : {
            "optim" : "SGD",
            "lr" : 0.005,
            "momentum" : 0.006,
            "epochs" : 35000,
            "weights_init" : "norm",
            "input_size" : 8,
            "hidden_size" : 7,
            "output_size" : 1,
        },
        
        "PSN" : {
            "optim" : "SGD",
            "lr" : 0.3,
            "momentum" : 0.5,
            "epochs" : 40000,
            "weights_init" : "norm",
            "input_size" : 7,
            "hidden_size" : 6,
            "output_size" : 1,
        }
    },
    
    "QQQ" : {
        "MLP" : {
            "optim" : "SGD",
            "lr" : 0.003,
            "momentum" : 0.005,
            "epochs" : 30000,
            "weights_init" : "norm",
            "input_size" : 7,
            "hidden_size" : 9,
            "output_size" : 1,
        },
        
        "RNN" : {
            "optim" : "SGD",
            "lr" : 0.002,
            "momentum" : 0.005,
            "epochs" : 35000,
            "weights_init" : "norm",
            "input_size" : 8,
            "hidden_size" : 10,
            "output_size" : 1,
        },
        
        "PSN" : {
            "optim" : "SGD",
            "lr" : 0.3,
            "momentum" : 0.4,
            "epochs" : 25000,
            "weights_init" : "norm",
            "input_size" : 9,
            "hidden_size" : 8,
            "output_size" : 1,
        }
    }
}