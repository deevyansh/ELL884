no_smoothing = {
    "method_name" : "NO_SMOOTH",
}

add_k = {
    "method_name" : "ADD_K",
    'k': 0.4
}

stupid_backoff = {
    "method_name" : "STUPID_BACKOFF",
    'alpha': 0.6
}

good_turing = {
    "method_name" : "GOOD_TURING",
}

interpolation = {
    "method_name" : "INTERPOLATION",
    'l': [0,0.5,0.3,0.2]
}

kneser_ney = {
    "method_name" : "KNESER_NEY",
    'd': 0.75
}

error_correction = {
    "internal_ngram_best_config" : {
        "method_name" : "STUPID_BACKOFF",
    },
}
