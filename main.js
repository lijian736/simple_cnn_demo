
import { Runtime } from "./src/runtime.js";

new Vue({
    el: '#app',
    data() {
        return {
            imagesList: [],
            imageSelected: '',
            isModelLoading: false,
            network: 'yolov1',
            //the downloader for resources
            downloader: null,
            downloadProgress: 0,
            inferProgress: 0,
            isInferring: false,
            //the runtime view
            runtime: null,
            webglAcce: true
        }
    },

    mounted() {
        //download the image list
        this.download('data/images/images.cfg', 'text').then((imagelist) => {
            let images = imagelist.split('\n');
            images.forEach((value, index, array) => {
                let imageinfo = value.split('=').map((line) => line.trim()).filter((line) => line.length > 0);
                array[index] = { name: imageinfo[0], url: imageinfo[1] };
            });

            this.imagesList = images;
        })

        this.runtime = new Runtime(document.querySelector("#canvas"));
    },

    methods: {
        /**
         * select the network
         * @param {string} networkName network name
         */
        selectNetwork(networkName) {
            this.network = networkName;
        },

        /**
         * select the image
         * @param {string} selectedImage the selected image name
         */
        imageSelChanged(selectedImage) {
            let result = this.imagesList.find((value) => {
                return value.name === selectedImage;
            });

            if (result) {
                this.runtime.loadImage(result.url);
            }
        },

        /**
         * load model weights data
         */
        loadModel() {
            this.isModelLoading = true;
            this.downloadModel({ net: 'data/models/' + this.network + '.cfg', weights: 'data/models/' + this.network + '.weights', labels: 'data/labels/' + this.network + '.txt' });
        },

        /**
         * download model definition, weights and labels
         * @param {Object} options download options
         */
        downloadModel(options) {

            this.download(options.net, 'text').then((cfg) => {
                this.runtime.networkcfg = cfg;
                return this.download(options.weights, 'arraybuffer');
            }).then((weights)=>{
                this.runtime.weights = weights;
                return this.download(options.labels, 'text');
            }).then((labels) => {
                this.runtime.labels = labels;

                this.isModelLoading = false;
                this.$message({ message: "Load model done", type: 'success', center: true, duration:1000 });
            });
        },

        /**
         * download
         * @param {string} url the url
         * @param {string} type the download type, 'arraybuffer' or 'text'
         */
        download(url, type) {
            return new Promise((resolve, reject) => {
                this.downloadProgress = 0;
                this.downloader = new XMLHttpRequest();
                this.downloader.open("GET", url, true);
                this.downloader.responseType = type;
                this.downloader.onload = () => {
                    const result = this.downloader.response;
                    if (result) {
                        resolve(result);
                    }
                };
                this.downloader.onabort = (e) => {
                    this.isModelLoading = false;
                    this.$message({ message: "Load aborted", type: 'warning', center: true, duration:1000 });
                };
                this.downloader.onerror = (e) => {
                    this.isModelLoading = false;
                    this.$message({ message: "Load error", type: 'error', center: true, duration:1000 });
                };
                this.downloader.ontimeout = (e) => {
                    this.isModelLoading = false;
                    this.$message({ message: "Load timeout", type: 'error', center: true, duration:1000 });
                };
                this.downloader.timeout = 1000 * 60 * 30; //30 minutes
                this.downloader.onprogress = (event) => {
                    this.downloadProgress = Math.min(Math.ceil(event.loaded / event.total * 100), 100);
                };

                this.downloader.send();
            });
        },

        /**
         * cancel the download
         */
        cancelDownload() {
            this.isModelLoading = false;
            this.downloader.abort();
        },

        /**
         * infer callback
         * @param {Object} status
         */
        inferCallback(status) {
            if (status.type === 'inferStatus') {
                this.inferProgress = Math.min(Math.ceil(status.current / status.total * 100), 100);
                if (status.current === status.total) {
                    this.isInferring = false;
                }
            } else if (status.type === 'noInput') {
                this.$message.error('No input image');
            } else if (status.type === 'noConfig') {
                this.$message.error('No network config')
            } else if (status.type === 'noWeights') {
                this.$message.error('No weights data')
            } else if (status.type === 'network') {
                if (!status.status) {
                    this.$message.error(status.info)
                }
            } else if (status.type === 'startInfer') {
                this.isInferring = true;
            } else if (status.type === 'weights'){
                console.log("%s - %s", status.type, status.info);
            } else {
                console.log("info: %s - %s", status.type, status.info);
            }
        },

        /**
         * start infer
         */
        startInfer() {
            this.runtime.startInfer(this.network, this.webglAcce, this.inferCallback);
        }
    }
})