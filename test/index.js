
window.it = (title, callback) => {
    console.log('%cstart test [%s]', 'color:blue',title);
    if (callback) {
        callback();
    }
}

window.describe = (title, callback) => {
    console.log('%c-----------start test group [%s]-----------', 'color:red', title);
    if (callback) {
        callback();
    }
    //new line
    console.log();
}