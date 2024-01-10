/**
 * post process
 */
export function postProcess(resnet50Network, topk) {
    let results = resnet50Network.output;
    let sortedResults = [];
    for(let i = 0; i < results.length; ++i){
        sortedResults.push({index: i, value: results[i]});
    }

    sortedResults.sort((ele1, ele2) => {
        let diff = ele1.value - ele2.value;
        if (diff < 0) {
            return 1;
        } else if (diff > 0) {
            return -1;
        } else {
            return 0;
        }
    });

    return sortedResults.slice(0, topk);
}