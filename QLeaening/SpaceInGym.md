# BOX -> R*n (x1, x2, x3, ..., xn)
This is a n-dimensional box, where each dimension is continuous.
```Python
    gym.spaces.Box(low = -10, high = 10, shape = (3, 3), dtype = np.float32)
```

# Discrete -> {0, 1, 2, ..., n-1}
```Python    
    gym.spaces.Discrete(n = 5)
```

# dictorionary -> {'position': Box(2,), 'velocity': Box(2,)}
```Python
    gym.spaces.Dict({'position': gym.spaces.Box(low = -10, high = 10, shape = (2,), dtype = np.float32),
                     'velocity': gym.spaces.Box(low = -1, high = 1, shape = (2,), dtype = np.float32)})
```

# Multi Binary -> [0, 1]^n (x1, x2, x3, ..., xn)
```Python
    gym.spaces.MultiBinary(n = 4)
```

# Multi Discrete -> {0, 1, 2, ..., n-1}^m (x1, x2, x3, ..., xm)
```Python
    gym.spaces.MultiDiscrete(nvec = [3, 3, 3])
```

# Tuple -> (Box(2,), Discrete(3))
```Python
    gym.spaces.Tuple((gym.spaces.Box(low = -10, high = 10, shape = (2,), dtype = np.float32),
                      gym.spaces.Discrete(n = 3)))
```

# prng -> Random Seed
```Python
    prng = np.random.RandomState(1)
```