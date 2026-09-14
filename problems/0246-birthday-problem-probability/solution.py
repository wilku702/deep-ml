import math

def birthday_problem(n: int, days: int = 365) -> float:
    """
    Calculate the probability that at least two people share the same birthday.
    
    Args:
        n: Number of people in the group
        days: Number of days in a year (default 365)
    
    Returns:
        float: Probability of at least one shared birthday, rounded to 4 decimal places
    """
    # Your code here
    
    if n <= 1:
        return 0.0
    elif n > days:
        return 1.0
    
    prob = math.factorial( days ) / ( math.factorial( days - n) * days**n)

    return round(1 - prob, 4)