import * as Utils from './utils.js'
import CNN from './network.js'

/**
 * post process
 * Yolov2 output: 
 *  5 * 19 * 19 * (80 + 4 + 1) total floats. 5 anchors, 19 * 19 grid cells, 80 classes, 4 coordinates, 1 confidence score
 *  the layout of the 5 anchors: Anchor 0 | Anchor 1 | Anchor 2 | Anchor 3 | Anchor 4
 * 
 * Each Anchor:
 * 19 * 19 * (80 + 4 + 1) total floats. 19 * 19 grid cells, 80 classes, 4 coordinates, 1 confidence score
 * the layout of an anchor: x | y | w | h | confidence | classes
 * x: 19 * 19 floats
 * y: 19 * 19 floats
 * w: 19 * 19 floats
 * h: 19 * 19 floats
 * confidence: 19 * 19 floats
 * classes: 19 * 19 * 80 floats
 * 
 * Classes: 
 * 19 * 19 * 80 floats.
 * the layout of the classes: class 0 | class 1 | class2 | ...... | class 79
 * each class has 19 * 19 floats
 *
 * @param {CNN.Network} yolov2Network the yolov2 network
 * @param {number} detectThreshold the threshold of object detection
 * @param {number} iouThreshold the threshold of IOU
 * @param {number} widthScale the input width scale
 * @param {number} heightScale the input height scale
 * 
 * @returns the bounding box arrayj
 */
export function postProcess(yolov2Network, detectThreshold, iouThreshold, widthScale, heightScale) {

    let entryIndex = (layer, location, entry) => {
        let n = Math.floor(location / (layer.outputWidth * layer.outputHeight));
        let loc = location % (layer.outputWidth * layer.outputHeight);
        return n * layer.outputWidth * layer.outputHeight * (layer.coords + layer.klasses + 1) + entry * layer.outputWidth * layer.outputHeight + loc;
    }

    let predictions = yolov2Network.output;

    let regionLyr = yolov2Network.tailLayer;
    const boxCount = regionLyr.boxCount;
    let boxes = [];
    for (let i = 0; i < boxCount; i++) {
        boxes.push(new CNN.BBox(regionLyr.classes));
    }

    //the grid count, 19x19
    const gridCount = regionLyr.outputWidth * regionLyr.outputHeight;

    for (let i = 0; i < gridCount; ++i) {
        //the grid row
        let row = Math.floor(i / regionLyr.outputWidth);
        //the grid column
        let col = i % regionLyr.outputWidth;

        //default is 5
        for (let n = 0; n < regionLyr.boxes; ++n) {
            let index = n * gridCount + i;
            for (let j = 0; j < regionLyr.classes; ++j) {
                boxes[index].prob[j] = 0;
            }

            let confidenceIndex = entryIndex(regionLyr, n * gridCount + i, regionLyr.coords);
            let boxIndex = entryIndex(regionLyr, n * gridCount + i, 0);
            let scale = predictions[confidenceIndex];
            //x range is [0, 1.0]
            boxes[index].x = (col + predictions[boxIndex]) / regionLyr.outputWidth;
            //y range is [0, 1.0]
            boxes[index].y = (row + predictions[boxIndex + gridCount]) / regionLyr.outputHeight;
            //w range is [0, 1.0]
            boxes[index].w = Math.exp(predictions[boxIndex + 2 * gridCount]) * regionLyr.biases[2 * n] / regionLyr.outputWidth;
            //h range is [0, 1.0]
            boxes[index].h = Math.exp(predictions[boxIndex + 3 * gridCount]) * regionLyr.biases[2 * n + 1] / regionLyr.outputHeight;
            boxes[index].objectness = scale > detectThreshold ? scale : 0;

            if (boxes[index].objectness > 0) {
                for (let j = 0; j < regionLyr.classes; ++j) {
                    let classIndex = entryIndex(regionLyr, n * gridCount + i, regionLyr.coords + 1 + j);
                    let prob = scale * predictions[classIndex];
                    boxes[index].prob[j] = (prob > detectThreshold) ? prob : 0;
                }
            }
        }
    }

    Utils.doNMS(boxes, boxCount, regionLyr.klasses, iouThreshold);

    boxes.forEach((value, index, array)=>{
        value.y /= heightScale;
        value.h /= heightScale;

        value.x /= widthScale;
        value.w /= widthScale;
    });
    return boxes;
}