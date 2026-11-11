def dedup(arr):
    seen = []
    result = []
    for x in arr:
        if x not in seen:
            seen.append(x)
            result.append(x)
    return result
