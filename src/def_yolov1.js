
import * as Utils from './utils.js'
import CNN from './network.js'

/**
* post process
* Yolov1 output: A | B | C
* A: grid * grid * classes, default: 7 * 7 * 20, 7 * 7 grids, and 20 classes probabilities
* B: grid * grid * boxes, default: 7 * 7 * 3, 7 * 7 grids, and 3 boxes probabilities
* C: grid * grid * boxes * coordinates, default: 7 * 7 * 3 * 4, 7 * 7 grids, 3 boxes, each box has 4 coordinates(x,y,w,h)
*
* @param {CNN.Network} yolov1Network the yolov1 network
* @param {number} detectThreshold the threshold of object detection
* @param {number} iouThreshold the threshold of IOU
* @param {number} widthScale the input width scale
* @param {number} heightScale the input height scale
* 
* @returns the bounding box array
*/
export function postProcess(yolov1Network, detectThreshold, iouThreshold, widthScale, heightScale) {
    let detLyr = yolov1Network.tailLayer;

    //7x7x3
    const boxCount = detLyr.boxCount;
    let boxes = [];
    for (let i = 0; i < boxCount; i++) {
        boxes.push(new CNN.BBox(detLyr.classes));
    }

    //the grid count, 7x7
    const gridCount = detLyr.side * detLyr.side;
    //all classes count, 7x7x20
    const klassesCount = gridCount * detLyr.classes;
    //all scale count, 7x7x3
    const scaleCount = gridCount * detLyr.num;

    let predictions = yolov1Network.output;

    for (let i = 0; i < gridCount; ++i) {
        //the grid row
        let row = Math.floor(i / detLyr.side);
        //the grid column
        let col = i % detLyr.side;

        for (let n = 0; n < detLyr.num; ++n) {
            let scaleIndex = klassesCount + i * detLyr.num + n;
            let scale = predictions[scaleIndex];

            let boxIndex = klassesCount + scaleCount + (i * detLyr.num + n) * 4;

            let index = i * detLyr.num + n;
            //x and y between 0 and 1
            boxes[index].x = (predictions[boxIndex + 0] + col) / detLyr.side;
            boxes[index].y = (predictions[boxIndex + 1] + row) / detLyr.side;
            //w and y between 0 and 1
            boxes[index].w = Math.pow(predictions[boxIndex + 2], (detLyr.sqrt ? 2 : 1));
            boxes[index].h = Math.pow(predictions[boxIndex + 3], (detLyr.sqrt ? 2 : 1));
            boxes[index].objectness = scale;

            const classIndex = i * detLyr.classes;
            for (let j = 0; j < detLyr.classes; ++j) {
                //compute the conditional probabilities
                let prob = scale * predictions[classIndex + j];
                boxes[index].prob[j] = (prob > detectThreshold) ? prob : 0;
            }
        }
    }

    Utils.doNMS(boxes, boxCount, detLyr.classes, iouThreshold);

    //normalize
    boxes.forEach((value, index, array) => {
        value.y /= heightScale;
        value.h /= heightScale;

        value.x /= widthScale;
        value.w /= widthScale;
    });
    return boxes;
}