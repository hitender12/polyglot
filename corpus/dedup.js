function dedup(arr) {
    let seen = [];
    let result = [];
    for (let i = 0; i < arr.length; i++) {
        let x = arr[i];
        if (seen.indexOf(x) === -1) {
            seen.push(x);
            result.push(x);
        }
    }
    return result;
}
