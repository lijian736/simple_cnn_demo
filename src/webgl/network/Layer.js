/**
 * Layer definition in Convolutional Neural Network
 */
export default class Layer {
    /**
     * constructor
     * @param {string} type the layer type
     * @param {Object} options the layers options 
     */
    constructor(type, options) {
        const {
            useWebGL = true
        } = options;

        this._type = type;
        this._useWebGL = useWebGL;

        //the workspace size for intermediate data
        this._workspaceSize = 0;
    }

    get type() {
        return this._type;
    }
    set type(newType) {
        this._type = newType;
    }

    get useWebGL(){
        return this._useWebGL;
    }
    set useWebGL(newValue){
        this._useWebGL = newValue;
    }

    get workspaceSize() {
        return this._workspaceSize;
    }
    set workspaceSize(newValue) {
        this._workspaceSize = newValue;
    }

    get info() {
        return "Layer-unknown";
    }

    forward(input, net) {
        throw new Error('layer ' + this.type + ' forward was not implemented.');
    }
}