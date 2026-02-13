const ort = require('onnxruntime-node');
const path = require('path');
const fs = require('fs').promises;

module.exports = function(RED) {
    function FavoriotMLNode(config) {
        RED.nodes.createNode(this, config);
        const node = this;
        let session = null;
        let preprocessorConfig = null;

        // Load preprocessor config from a specific path
        async function loadPreprocessorConfig(configPath) {
            try {
                node.log(`Attempting to load preprocessor config from: ${configPath}`);
                
                const configContent = await fs.readFile(configPath, 'utf8');
                preprocessorConfig = JSON.parse(configContent);
                
                // CRITICAL: Normalize ALL keys/values with whitespace trimming
                const normalizeString = s => (typeof s === 'string' ? s.trim() : s);
                const normalizeObjectKeys = (obj) => {
                    if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
                        const normalized = {};
                        for (const [key, value] of Object.entries(obj)) {
                            normalized[normalizeString(key)] = normalizeObjectKeys(value);
                        }
                        return normalized;
                    } else if (Array.isArray(obj)) {
                        return obj.map(normalizeObjectKeys);
                    }
                    return obj;
                };
                
                preprocessorConfig = normalizeObjectKeys(preprocessorConfig);
                
                // Normalize arrays of feature names
                const normalizeArray = (arr) => arr?.map?.(s => normalizeString(s)) || [];
                preprocessorConfig.feature_order = normalizeArray(preprocessorConfig.feature_order);
                preprocessorConfig.categorical_features = normalizeArray(preprocessorConfig.categorical_features);
                preprocessorConfig.numeric_features = normalizeArray(preprocessorConfig.numeric_features);
                
                node.log(`✓ Preprocessor config loaded: ${path.basename(configPath)}`);
                return true;
            } catch (e) {
                if (e.code !== 'ENOENT') {
                    node.warn(`Preprocessor config warning: ${e.message}`);
                }
                // Fallback to empty config if file is missing
                preprocessorConfig = {
                    feature_order: [],
                    categorical_features: [],
                    categorical_encoders: {},
                    scaler: null
                };
                node.log('⚠ No preprocessor config found - numeric inputs only');
                return false;
            }
        }

        function encodeCategoricalInputs(payload) {
            if (!preprocessorConfig?.categorical_encoders) return payload;

            const encoded = { ...payload };
            for (const [feat, encInfo] of Object.entries(preprocessorConfig.categorical_encoders)) {
                if (feat in encoded && typeof encoded[feat] === 'string') {
                    const strValue = encoded[feat].trim() || 'MISSING';
                    const mapping = encInfo.mapping || {};
                    
                    if (strValue in mapping) {
                        encoded[feat] = mapping[strValue];
                    } else {
                        // Case-insensitive fallback
                        const lowerValue = strValue.toLowerCase();
                        const caseInsensitiveMatch = Object.keys(mapping).find(
                            key => key.toLowerCase() === lowerValue
                        );
                        
                        if (caseInsensitiveMatch) {
                            encoded[feat] = mapping[caseInsensitiveMatch];
                        } else {
                            // Use __UNKNOWN__ key or highest index
                            const unknownIdx = mapping['__UNKNOWN__'] ?? 
                                              (Object.values(mapping).length > 0 
                                                ? Math.max(...Object.values(mapping)) + 1 
                                                : 0);
                            encoded[feat] = unknownIdx;
                            node.warn(`⚠ Unknown category: ${feat}='${strValue}' → ${unknownIdx}`);
                        }
                    }
                }
            }
            return encoded;
        }

        function prepareInputTensor(payload) {
            const encodedPayload = encodeCategoricalInputs(payload);
            let features;

            if (preprocessorConfig?.feature_order?.length > 0) {
                features = preprocessorConfig.feature_order.map(feat => {
                    if (!(feat in encodedPayload)) {
                        throw new Error(`Missing feature '${feat}'. Expected: ${preprocessorConfig.feature_order.join(', ')}`);
                    }
                    const val = parseFloat(encodedPayload[feat]);
                    if (isNaN(val)) {
                        throw new Error(`Non-numeric after encoding: ${feat}='${encodedPayload[feat]}'`);
                    }
                    return val;
                });
            } else {
                features = Object.values(encodedPayload).map(v => {
                    const val = parseFloat(v);
                    if (isNaN(val)) {
                        throw new Error(`Non-numeric input: ${v} (no preprocessor config loaded)`);
                    }
                    return val;
                });
            }

            if (preprocessorConfig?.scaler?.type === 'standard') {
                const { mean, scale } = preprocessorConfig.scaler;
                if (mean?.length === features.length && scale?.length === features.length) {
                    features = features.map((val, i) => (val - mean[i]) / scale[i]);
                }
            }

            return new ort.Tensor('float32', Float32Array.from(features), [1, features.length]);
        }

        async function loadModel() {
            try {
                const modelPath = path.resolve(config.onnxPath);
                // FIX: Prioritize config.preprocessorPath from UI, fallback to auto-replace
                const configPath = config.preprocessorPath ? 
                                   path.resolve(config.preprocessorPath) : 
                                   modelPath.replace(/\.onnx$/, '.json');

                node.status({ fill: "yellow", shape: "ring", text: "Loading..." });
                await fs.access(modelPath);
                
                // Pass the specific path to the loader
                await loadPreprocessorConfig(configPath);
                
                session = await ort.InferenceSession.create(modelPath, {
                    executionProviders: ['cpu']
                });

                node.status({ fill: "green", shape: "dot", text: "Ready" });
            } catch (e) {
                node.error(`Model load failed: ${e.message}`);
                node.status({ fill: "red", shape: "ring", text: "Load Error" });
            }
        }

        node.on('input', async function(msg) {
            if (!session) {
                node.error("ONNX session not ready.");
                return;
            }

            try {
                node.status({ fill: "blue", shape: "dot", text: "Inferring..." });
                const inputTensor = prepareInputTensor(msg.payload);
                const inputName = session.inputNames[0];
                const results = await session.run({ [inputName]: inputTensor });

                let prediction = results.label?.data[0] || results[session.outputNames[0]]?.data[0];
                if (typeof prediction === 'number' && !Number.isInteger(prediction)) {
                    prediction = Math.round(prediction);
                }

                msg.payload = {
                    prediction: prediction,
                    status: "success",
                    timestamp: new Date().toISOString()
                };

                node.status({ fill: "green", shape: "dot", text: "Ready" });
                node.send(msg);
            } catch (e) {
                node.error(`Inference failed: ${e.message}`);
                node.status({ fill: "red", shape: "dot", text: "Error" });
                node.send({ payload: { error: e.message, status: "error" } });
            }
        });

        node.on('close', () => {
            session = null;
            preprocessorConfig = null;
        });

        if (config.onnxPath) {
            loadModel();
        }
    }
    RED.nodes.registerType('favoriot-ml', FavoriotMLNode);
};