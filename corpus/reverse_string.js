function reverseString(s) {
    let result = "";
    for (let i = 0; i < s.length; i++) {
        let ch = s[i];
        result = ch + result;
    }
    return result;
}
