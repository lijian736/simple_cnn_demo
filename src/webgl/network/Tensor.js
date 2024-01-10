
/**
 * The tensor object with data and data shapes
 */
export default class Tensor {
    /**
     * constructor
     * @param {Object} options including data and shape, type is Array
     */
    constructor(options={}) {
        const {
            data = [],
            shape = [],
        } = options;

        this._data = data;
        this._shape = shape;
    }

    get data(){
        return this._data;
    }

    get shape(){
        return this._shape;
    }
}