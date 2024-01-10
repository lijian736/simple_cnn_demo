import vertexShaderSource from './shader/vertex_shader.js'

/**
 * WebGL2 utils
 */
class WebGL2Utils {
    constructor() {
    }

    /**
     * Create a shader from source
     * @param {WebGL2RenderingContext} gl The WebGL2RenderingContext
     * @param {string} shaderSource The shader source
     * @param {number} shaderType The shader type
     * @param {FunctionStringCallback} errorCallback The error callback function
     * @returns {WebGLShader} The created WebGLShader
     */
    static createShaderFromSource(gl, shaderSource, shaderType, errorCallback) {
        //the error callback function
        const errorFunc = errorCallback || console.error;
        //Create the shader object
        const shader = gl.createShader(shaderType);
        if (!shader) {
            errorFunc('create WebGL shader failed');
        }
        //Load the shader source
        gl.shaderSource(shader, shaderSource);
        //Compile the shader
        gl.compileShader(shader);
        //Check the compile status
        const status = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
        if (!status) {
            //Compilation error occurs, retrieve the error information
            const lastError = gl.getShaderInfoLog(shader);
            const errorInfo = 'Error occurs when compile shader \'' + shader + '\':' + lastError + '\n' + shaderSource.split('\n').map((line, index) => `${index + 1}: ${line}`).join('\n');
            errorFunc(errorInfo);
            gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    /**
     * Create a program. Attaches shaders, links the program
     * @param {WebGL2RenderingContext} gl The WebGL2RenderingContext
     * @param {WebGLShader[]} shaders The WebGL shaders array to be attached
     * @param {FunctionStringCallback} errorCallback The error callback function
     * @returns {WebGLProgram} The created WebGLProgram
     */
    static createProgram(gl, shaders, errorCallback) {
        //the error callback function
        const errorFunc = errorCallback || console.error;
        //Create the gl program
        const program = gl.createProgram();
        if (!program) {
            errorFunc('create WebGL program failed');
        }
        //Attach the shaders
        shaders.forEach((shader) => {
            gl.attachShader(program, shader);
        });

        //Link the program
        gl.linkProgram(program);

        //Check the link status
        const status = gl.getProgramParameter(program, gl.LINK_STATUS);
        if (!status) {
            //Link error occurs, retrieve the error information
            const lastError = gl.getProgramInfoLog(program);
            const errorInfo = 'Error occurs when link program \'' + program + '\':' + lastError;
            errorFunc(errorInfo);
            gl.deleteProgram(program);
            return null;
        }

        return program;
    }

    /**
     * Set up the plain rect vertices for the program
     * @param {WebGL2RenderingContext} gl The WebGL2RenderingContext
     * @param {WebGLProgram} program WebGLProgram
     * @param {FunctionStringCallback} errorCallback The error callback function
     * @returns the [PositionBuffer, TextureBuffer, IndicesBuffer] array
     */
    static setupPlainVertices(gl, program, errorCallback) {
        //the error callback function
        const errorFunc = errorCallback || console.error;

        //Get position of 'position' in the vertex shader
        const positionLocation = gl.getAttribLocation(program, 'inPosition');
        if (positionLocation < 0) {
            errorFunc("can NOT find 'inPosition' in the program");
        }

        //create buffer for vertices and bind to vertex shader
        const positionBuffer = gl.createBuffer();
        if (!positionBuffer) {
            errorFunc("can NOT create position buffer in the program");
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        //left bottom -> right bottom -> right top -> left top
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]), gl.STATIC_DRAW);
        gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(positionLocation);

        //create buffer for texture mapping
        const texcoordLocation = gl.getAttribLocation(program, 'inTexCoord');
        if (texcoordLocation < 0) {
            errorFunc("can NOT find 'inTexCoord' in the program");
        }

        //create buffer for textures and bind to vertex shader
        const texcoordBuffer = gl.createBuffer();
        if (!texcoordBuffer) {
            errorFunc("can NOT create texture buffer in the program");
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
        //left bottom -> right bottom -> right top -> left top
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]), gl.STATIC_DRAW);
        gl.vertexAttribPointer(texcoordLocation, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(texcoordLocation);;

        //create buffer for indices and bind to vertex shader
        const indicesBuffer = gl.createBuffer();
        if (!indicesBuffer) {
            errorFunc("can NOT create indices buffer in the program");
        }

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indicesBuffer);
        //draw 2 triangles
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);

        return [positionBuffer, texcoordBuffer, indicesBuffer];
    }

    /**
     * Create a 2d texture
     * @param {WebGL2RenderingContext} gl The WebGL2RenderingContext
     * @param {number} width the texture width
     * @param {number} height the texture height
     * @param {Float32Array} data the texture data
     * @param {FunctionStringCallback} errorCallback The error callback function
     * @returns the created texture
     */
    static create2DTexture(gl, width, height, data, errorCallback) {
        //the error callback function
        const errorFunc = errorCallback || console.error;

        //Create a texture
        let texture = gl.createTexture();
        if (!texture) {
            errorFunc("can NOT create texture");
        }
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);

        //Set the texture parameters
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        return texture;
    }

    /**
    * Create a 2d array texture
    * @param {WebGL2RenderingContext} gl The WebGL2RenderingContext
    * @param {number} width the texture width
    * @param {number} height the texture height
    * @param {number} depth the texture depth
    * @param {Float32Array} data the texture data
    * @param {FunctionStringCallback} errorCallback The error callback function
    * @returns the created texture
    */
    static create2DArrayTexture(gl, width, height, depth, data, errorCallback) {
        //the error callback function
        const errorFunc = errorCallback || console.error;

        //Create a texture
        let texture = gl.createTexture();
        if (!texture) {
            errorFunc("can NOT create texture");
        }
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
        gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.R32F, width, height, depth, 0, gl.RED, gl.FLOAT, data);

        //Set the texture parameters
        //Clamp to edge, no clamp to border
        gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        //disable interpolation
        gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        return texture;
    }

    /**
     * Create a 2d texture list
     * @param {WebGL2RenderingContext} gl The WebGL2RenderingContext
     * @param {Object} options the texture options
     * @param {FunctionStringCallback} errorCallback The error callback function
     * @returns the created texture
     */
    static create2DTextureList(gl, options, errorCallback) {
        //the error callback function
        const errorFunc = errorCallback || console.error;

        //the inner create texture function
        const createTextureInner = (data, width, height) => {
            //Create a texture
            let texture = gl.createTexture();
            if (!texture) {
                errorFunc("can NOT create texture list");
            }
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);

            //Set the texture parameters
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

            return texture;
        };

        let results = [];

        //Get width fragmented textures
        if (options.widthFragmented) {
            const numTextures = Math.ceil(options.width / options.MAX_TEXTURE_SIZE);

            //the textures
            for (let i = 0; i < numTextures; i++) {
                if (i == numTextures - 1) {
                    //the remaining texture width
                    const width = options.width - options.MAX_TEXTURE_SIZE * i;
                    //the start position in width
                    const start = options.MAX_TEXTURE_SIZE * i;

                    //the texture data
                    let data = new Float32Array(width * options.height);
                    let index = 0;
                    for (let j = 0; j < options.height; j++) {
                        for (let k = 0; k < width; k++) {
                            data[index] = options.data[options.width * j + start + k];
                            index++;
                        }
                    }

                    //create the texture
                    let texture = createTextureInner(data, options.width - options.MAX_TEXTURE_SIZE * i, options.height);
                    results.push(texture);
                } else {
                    //the texture width
                    const width = options.MAX_TEXTURE_SIZE;
                    //the start position in width
                    const start = options.MAX_TEXTURE_SIZE * i;

                    //the texture data
                    let data = new Float32Array(width * options.height);
                    let index = 0;
                    for (let j = 0; j < options.height; j++) {
                        for (let k = 0; k < width; k++) {
                            data[index] = options.data[options.width * j + start + k];
                            index++;
                        }
                    }

                    //create the texture
                    let texture = createTextureInner(data, options.MAX_TEXTURE_SIZE, options.height);
                    results.push(texture);
                }
            }
        } else if (options.heightFragmented) {
            const numTextures = Math.ceil(options.height / options.MAX_TEXTURE_SIZE);
            for (let i = 0; i < numTextures; i++) {
                if (i == numTextures - 1) {
                    let data = options.data.subarray(options.MAX_TEXTURE_SIZE * options.width * i);
                    let texture = createTextureInner(data, options.width, options.height - options.MAX_TEXTURE_SIZE * i);
                    results.push(texture);
                } else {
                    let data = options.data.subarray(options.MAX_TEXTURE_SIZE * options.width * i, options.MAX_TEXTURE_SIZE * options.width * (i + 1));
                    let texture = createTextureInner(data, options.width, options.MAX_TEXTURE_SIZE);
                    results.push(texture);
                }
            }
        }

        return results;
    }
};

/**
 * The WebGL2 class. It runs only in web browser context
 */
class WebGL2 {
    constructor() {
        //the web browser window object
        if (window) {
            this._canvas = window.document.createElement('canvas');
            this._webgl2 = this._canvas.getContext('webgl2');
            if (this._webgl2) {
                this._supported = true;
                this._webgl2.getExtension('EXT_color_buffer_float');
                //the texture max size
                this._textureSize = this._webgl2.getParameter(this._webgl2.MAX_TEXTURE_SIZE)
                //the textures number supported
                this._textureUnits = this._webgl2.getParameter(this._webgl2.MAX_TEXTURE_IMAGE_UNITS)
                //the 3D texture depth max size
                this._3dtextureSize = this._webgl2.getParameter(this._webgl2.MAX_3D_TEXTURE_SIZE);
                //the array texture size
                this._arrayTextureSize = this._webgl2.getParameter(this._webgl2.MAX_ARRAY_TEXTURE_LAYERS);
                //the maximum vertex uniform components
                this._vertex_uniform_components = this._webgl2.getParameter(this._webgl2.MAX_VERTEX_UNIFORM_COMPONENTS);
                //the maximum vertex uniform vectors
                this._vertex_uniform_vectors = this._webgl2.getParameter(this._webgl2.MAX_VERTEX_UNIFORM_VECTORS);
                //the maximum fragment uniform components
                this._fragment_uniform_components = this._webgl2.getParameter(this._webgl2.MAX_FRAGMENT_UNIFORM_COMPONENTS);
                //the maximum fragment uniform vectors
                this._fragment_uniform_vectors = this._webgl2.getParameter(this._webgl2.MAX_FRAGMENT_UNIFORM_VECTORS);
                //initialize the context
                this._initialize();
            } else {
                throw new Error('WebGL2 is NOT supported, please check the runtime context');
            }
        } else {
            throw new Error('window is NOT supported, please check the runtime context');
        }

        //the vertex buffers
        this._vertexBuffers = [];
        //the texture buffers
        this._textureBuffers = [];
        //the index buffers
        this._indicesBuffers = [];
        //the textures
        this._textures = [];
    }

    get MAX_TEXTURE_SIZE() {
        return this._textureSize;
    }

    get MAX_TEXTURE_IMAGE_UNITS() {
        return this._textureUnits;
    }

    get MAX_3D_TEXTURE_SIZE() {
        return this._3dtextureSize;
    }

    get MAX_ARRAY_TEXTURE_LAYERS() {
        return this._arrayTextureSize;
    }

    get MAX_VERTEX_UNIFORM_COMPONENTS() {
        return this._vertex_uniform_components;
    }

    get MAX_VERTEX_UNIFORM_VECTORS() {
        return this._vertex_uniform_vectors;
    }

    get MAX_FRAGMENT_UNIFORM_COMPONENTS() {
        return this._fragment_uniform_components;
    }

    get MAX_FRAGMENT_UNIFORM_VECTORS() {
        return this._fragment_uniform_vectors;
    }

    /**
     * initialize the context
     */
    _initialize() {
        this._vertexShader = WebGL2Utils.createShaderFromSource(this._webgl2, vertexShaderSource, this._webgl2.VERTEX_SHADER, (errorInfo) => {
            this._supported = false
            throw new Error(errorInfo);
        });
    }

    /**
     * create a WebGL program with the default vertex shader and the specified fragment source
     * @param {string} fragmentSoruce the fragment shader source
     * @returns the program
     */
    createProgram(fragmentSoruce) {
        let fragmentShader = WebGL2Utils.createShaderFromSource(this._webgl2, fragmentSoruce, this._webgl2.FRAGMENT_SHADER, (errorInfo) => {
            this._supported = false
            throw new Error(errorInfo);
        });

        let program = WebGL2Utils.createProgram(this._webgl2, [this._vertexShader, fragmentShader], (errorInfo) => {
            this._supported = false
            throw new Error(errorInfo);
        });

        let [positionBuffer, textureBuffer, indicesBuffer] = WebGL2Utils.setupPlainVertices(this._webgl2, program, (errorInfo) => {
            this._supported = false
            throw new Error(errorInfo);
        });

        this._vertexBuffers.push(positionBuffer);
        this._textureBuffers.push(textureBuffer);
        this._indicesBuffers.push(indicesBuffer);

        return program;
    }

    /**
     * create WebGL 2D texture
     * @param {Object} options
     * @returns the texture
     */
    create2DTexture(options = {}) {
        const {
            textureWidth = undefined,
            textureHeight = undefined,
            textureData = undefined,
        } = options;

        if (textureWidth > this.MAX_TEXTURE_SIZE || textureHeight > this.MAX_TEXTURE_SIZE) {
            throw new Error('Texture size exceeds the max size');
        }

        let texture = WebGL2Utils.create2DTexture(this._webgl2, textureWidth, textureHeight, textureData, (errorInfo) => {
            throw new Error(errorInfo);
        });

        this._textures.push(texture);
        return texture;
    }

    /**
     * create WebGL 2D Array texture
     * @param {Object} options 
     * @returns the 2D Array texture
     */
    create2DArrayTexture(options = {}) {
        const {
            textureWidth = undefined,
            textureHeight = undefined,
            textureDepth = undefined,
            textureData = undefined,
        } = options;

        if (textureWidth > this.MAX_TEXTURE_SIZE || textureHeight > this.MAX_TEXTURE_SIZE || textureDepth > this.MAX_ARRAY_TEXTURE_LAYERS) {
            throw new Error('Texture 2d array size exceeds the max size');
        }

        let texture = WebGL2Utils.create2DArrayTexture(this._webgl2, textureWidth, textureHeight, textureDepth, textureData, (errorInfo) => {
            throw new Error(errorInfo);
        });

        this._textures.push(texture);
        return texture;
    }

    /**
     * use the program
     * @param {WebGLProgram} program the webgl program
     */
    useProgram(program) {
        this._webgl2.useProgram(program);
    }

    /**
     * Bind the program's uniforms
     * @param {WebGLProgram} program 
     * @param {Array} uniforms uniform array
     */
    bindUniforms(program, uniforms) {
        uniforms.forEach((value, index, array) => {
            let name = value.name;
            let type = value.type;
            let data = value.data;

            const location = this._webgl2.getUniformLocation(program, name);
            if (type === 'int' || type === 'bool') {
                this._webgl2.uniform1i(location, data);
            } else if (type === 'float') {
                this._webgl2.uniform1f(location, data);
            }
        });
    }

    /**
   * Bind input textures within program, handling inputs that are fragmented
   *
   * @param {WebGLProgram} program the program
   * @param {Object[]} inputs the textures input
   */
    bindInputTextures(program, inputs) {
        inputs.forEach(({ texture, name }, index) => {
            this._webgl2.activeTexture(this._webgl2.TEXTURE0 + index)
            this._webgl2.bindTexture(this._webgl2.TEXTURE_2D, texture)
            this._webgl2.uniform1i(this._webgl2.getUniformLocation(program, name), index)
        });
    }

    /**
     * Bind output texture
     *
     * @param {WebGLTexture} outputTexture
     * @param {number} width the texture width
     * @param {number} height the textue height
     */
    bindOutputTexture(outputTexture, width, height) {
        this._webgl2.viewport(0, 0, width, height);
        this._framebuffer = this._framebuffer || this._webgl2.createFramebuffer();
        this._webgl2.bindFramebuffer(this._webgl2.FRAMEBUFFER, this._framebuffer);
        this._webgl2.framebufferTexture2D(this._webgl2.FRAMEBUFFER, this._webgl2.COLOR_ATTACHMENT0, this._webgl2.TEXTURE_2D, outputTexture, 0);
    }

    /**
     * Run the WebGL's program
     * @param {Object} program the program
     * @param {Array} uniforms the uniform array
     * @param {Array} inputs the inputs
     * @param {Float32Array} output the output
     * 
     */
    run(options) {
        const {
            program,
            uniforms,
            inputs,
            output
        } = options;

        this.useProgram(program);
        this.bindUniforms(program, uniforms)

        //let outTextureData = new Float32Array(output.height * output.width);
        let outputTexture = webgl2.create2DTexture({ textureWidth: output.width, textureHeight: output.height, textureData: null });

        this.bindOutputTexture(outputTexture, output.width, output.height);
        this.bindInputTextures(program, inputs);
        
        this._webgl2.drawElements(this._webgl2.TRIANGLES, 6, this._webgl2.UNSIGNED_SHORT, 0);

        return this.readDataFromTexture(output.width, output.height);
    }

    /**
     * read data from output texture
     * @param {number} width the texture width
     * @param {number} height the texture height
     * @returns Float32Array
     */
    readDataFromTexture(width, height) {
        const buf = new Float32Array(height * width);
        this._webgl2.readPixels(0, 0, width, height, this._webgl2.RED, this._webgl2.FLOAT, buf);
        return buf;
    }

    /**
     * release the resources
     */
    release() {
        this._vertexBuffers.forEach((buffer) => this._webgl2.deleteBuffer(buffer));
        this._textureBuffers.forEach((buffer) => this._webgl2.deleteBuffer(buffer));
        this._indicesBuffers.forEach((buffer) => this._webgl2.deleteBuffer(buffer));
        this._textures.forEach((texture) => this._webgl2.deleteTexture(texture));

        this._vertexBuffers = [];
        this._textureBuffers = [];
        this._indicesBuffers = [];
        this._textures = [];
    }
};

const webgl2 = new WebGL2();

export { webgl2 }