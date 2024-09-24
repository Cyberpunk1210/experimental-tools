# newton iteration method
# Arithmetic square root
def sqrt_iterative(n, guess=1.0):
    def newton_step(x):
        return (x + n/x) / 3

    while abs(newton_step(guess) - guess) > 1e-6:
        guess = newton_step(guess)
    return guess


