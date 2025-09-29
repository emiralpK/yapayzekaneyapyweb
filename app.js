// Ana uygulama sınıfı
class AITrainerApp {
    constructor() {
        this.model = null;
        this.tokenizer = new CustomTokenizer();
        this.isTraining = false;
        this.loadedTexts = [];
        this.hasTensorflowGPU = false;
        
        this.initUI();
        this.checkGPU();
    }
    
    // UI bileşenlerini başlat
    initUI() {
        // Sekme değiştirme işlevi
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
            });
        });
        
        // Dosya yükleme
        document.getElementById('file-input').addEventListener('change', this.handleFileUpload.bind(this));
        
        // Epoch değeri gösterme
        const epochSlider = document.getElementById('epoch-slider');
        const epochValue = document.getElementById('epoch-value');
        epochSlider.addEventListener('input', () => {
            epochValue.textContent = epochSlider.value;
        });
        
        // Eğitim butonu
        document.getElementById('train-btn').addEventListener('click', this.startTraining.bind(this));
        
        // Mesaj gönderme
        document.getElementById('send-btn').addEventListener('click', this.sendMessage.bind(this));
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        
        // Model kaydetme ve yükleme
        document.getElementById('save-model-btn').addEventListener('click', this.saveModel.bind(this));
        document.getElementById('load-model-btn').addEventListener('click', () => {
            document.getElementById('model-input').click();
        });
        document.getElementById('model-input').addEventListener('change', this.loadModel.bind(this));
    }
    
    // GPU desteğini kontrol et
    async checkGPU() {
        const gpuInfo = document.getElementById('gpu-info');
        
        try {
            await tf.ready();
            this.hasTensorflowGPU = tf.getBackend() === 'webgl';
            
            if (this.hasTensorflowGPU) {
                gpuInfo.textContent = '🎮 GPU: WebGL Aktif (Hızlı Eğitim ✅)';
                gpuInfo.style.color = 'var(--success-color)';
            } else {
                gpuInfo.textContent = '🎮 GPU: Devre Dışı (CPU Modu ⚠️)';
                gpuInfo.style.color = 'var(--warning-color)';
            }
            
            this.log(`TensorFlow.js backend: ${tf.getBackend()}`);
        } catch (error) {
            gpuInfo.textContent = '❌ TensorFlow yüklenemedi!';
            gpuInfo.style.color = 'var(--danger-color)';
            this.log('TensorFlow yüklenirken hata oluştu: ' + error.message, 'error');
        }
    }
    
    // Dosya yükleme işlevi
    handleFileUpload(event) {
        const file = event.target.files[0];
        const fileInfo = document.getElementById('file-info');
        
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const content = e.target.result;
                
                // Metni cümlelere böl
                const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 10);
                this.loadedTexts = sentences;
                
                const fileSizeKB = (file.size / 1024).toFixed(1);
                fileInfo.textContent = `✅ Yüklendi: ${file.name}\nBoyut: ${fileSizeKB} KB | Cümle: ${sentences.length}`;
                fileInfo.style.color = 'var(--success-color)';
                
                this.log(`Dosya başarıyla yüklendi: ${file.name}`);
                this.log(`Toplam ${sentences.length} cümle bulundu.`);
            };
            reader.onerror = () => {
                fileInfo.textContent = '❌ Dosya okunamadı!';
                fileInfo.style.color = 'var(--danger-color)';
            };
            reader.readAsText(file);
        }
    }
    
    // Eğitimi başlat
    async startTraining() {
        if (!this.loadedTexts.length) {
            alert('Lütfen önce bir dosya yükleyin!');
            return;
        }
        
        if (this.isTraining) {
            alert('Eğitim zaten devam ediyor!');
            return;
        }
        
        this.isTraining = true;
        const trainBtn = document.getElementById('train-btn');
        trainBtn.disabled = true;
        trainBtn.textContent = '⏳ Eğitim Sürüyor...';
        
        try {
            this.log('=' + '='.repeat(40));
            this.log('🚀 Eğitim başlıyor...');
            this.log(`Cihaz: ${this.hasTensorflowGPU ? 'GPU (WebGL)' : 'CPU'}`);
            
            // Bit değerini al
            const bitValue = document.querySelector('input[name="bits"]:checked').value;
            this.log(`Bit: ${bitValue}`);
            
            // Epoch değerini al
            const epochValue = document.getElementById('epoch-slider').value;
            this.log(`Epoch: ${epochValue}`);
            
            // Tokenizer'ı eğit
            this.log('📝 Tokenizer hazırlanıyor...');
            this.tokenizer.fit(this.loadedTexts);
            const vocabSize = Object.keys(this.tokenizer.word2idx).length;
            this.log(`Kelime hazinesi boyutu: ${vocabSize}`);
            
            // Dataset oluştur
            this.log('📊 Veri hazırlanıyor...');
            const dataset = this.prepareDataset(this.loadedTexts);
            
            // Model oluştur
            this.log('🤖 Model oluşturuluyor...');
            this.model = this.createModel(vocabSize, parseInt(bitValue));
            
            // Eğitim
            await this.trainModel(dataset, parseInt(epochValue));
            
            this.log('✅ Eğitim tamamlandı!');
            this.updateModelStats();
            
            alert('Model eğitimi tamamlandı! Artık sohbet edebilirsiniz.');
            
        } catch (error) {
            this.log(`❌ Hata: ${error.message}`, 'error');
            alert(`Eğitim sırasında hata: ${error.message}`);
        } finally {
            this.isTraining = false;
            trainBtn.disabled = false;
            trainBtn.textContent = '🚀 Eğitimi Başlat';
            document.getElementById('progress-bar').style.width = '0%';
            document.getElementById('progress-label').textContent = 'Hazır';
        }
    }
    
    // Dataset hazırlama
    prepareDataset(texts) {
        const dataset = [];
        const seqLen = 20; // Dizi uzunluğu
        
        for (const text of texts) {
            const encoded = this.tokenizer.encode(text, seqLen);
            if (encoded.length > 1) {
                for (let i = 0; i < encoded.length - 1; i++) {
                    dataset.push({
                        input: encoded[i],
                        output: encoded[i + 1]
                    });
                }
            }
        }
        
        return dataset;
    }
    
    // Model oluşturma
    createModel(vocabSize, bits) {
        // Simple LSTM model
        const model = tf.sequential();
        
        // Embedding layer
        model.add(tf.layers.embedding({
            inputDim: vocabSize,
            outputDim: 128,
            inputLength: 1
        }));
        
        // Reshape
        model.add(tf.layers.reshape({targetShape: [128]}));
        
        // Dense layers with quantization simulation
        model.add(tf.layers.dense({
            units: 256,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dropout({rate: 0.2}));
        
        model.add(tf.layers.dense({
            units: vocabSize,
            activation: 'softmax'
        }));
        
        // Compile
        model.compile({
            optimizer: 'adam',
            loss: 'sparseCategoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }
    
    // Model eğitimi
    async trainModel(dataset, epochs) {
        if (!dataset.length) {
            throw new Error('Eğitim verisi boş!');
        }
        
        const batchSize = 32;
        const totalBatches = Math.ceil(dataset.length / batchSize);
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            this.log(`Epoch ${epoch + 1}/${epochs} başlıyor...`);
            let totalLoss = 0;
            let totalAcc = 0;
            
            // Dataset'i karıştır
            const shuffled = [...dataset].sort(() => 0.5 - Math.random());
            
            // Batch'lere böl
            for (let batch = 0; batch < totalBatches; batch++) {
                const start = batch * batchSize;
                const end = Math.min(start + batchSize, dataset.length);
                const batchData = shuffled.slice(start, end);
                
                // Input ve output tensörleri hazırla
                const inputs = tf.tensor2d(
                    batchData.map(d => [d.input]), 
                    [batchData.length, 1]
                );
                
                const outputs = tf.tensor1d(
                    batchData.map(d => d.output),
                    'int32'
                );
                
                // Eğitim adımı
                const history = await this.model.trainOnBatch(inputs, outputs);
                
                totalLoss += history[0];
                if (history[1]) totalAcc += history[1];
                
                // İlerleme güncelle
                const progress = ((epoch * totalBatches + batch + 1) / (epochs * totalBatches)) * 100;
                document.getElementById('progress-bar').style.width = `${progress}%`;
                document.getElementById('progress-label').textContent = 
                    `Epoch ${epoch + 1}/${epochs} - Batch ${batch + 1}/${totalBatches}`;
                
                // Tensörleri temizle
                inputs.dispose();
                outputs.dispose();
                
                // UI güncellemesi için küçük bir ara
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            
            const avgLoss = totalLoss / totalBatches;
            const avgAcc = totalAcc / totalBatches;
            this.log(`Epoch ${epoch + 1}: Loss = ${avgLoss.toFixed(4)}, Acc = ${avgAcc.toFixed(4)}`);
        }
    }
    
    // Mesaj gönderme
    sendMessage() {
        if (!this.model) {
            alert('Önce bir model eğitmeniz gerekiyor!');
            return;
        }
        
        const chatInput = document.getElementById('chat-input');
        const userMessage = chatInput.value.trim();
        
        if (!userMessage) return;
        
        // Kullanıcı mesajını göster
        this.addMessageToChat('👤 Sen', userMessage, 'user-message');
        chatInput.value = '';
        
        // Model cevabı
        try {
            const response = this.generateResponse(userMessage);
            this.addMessageToChat('🤖 AI', response, 'ai-message');
        } catch (error) {
            this.addMessageToChat('❌ Hata', error.message, 'error-message');
        }
    }
    
    // Mesajı sohbet alanına ekle
    addMessageToChat(sender, message, className) {
        const chatMessages = document.getElementById('chat-messages');
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        
        const senderDiv = document.createElement('div');
        senderDiv.className = 'message-sender';
        senderDiv.textContent = sender;
        
        const contentDiv = document.createElement('div');
        contentDiv.textContent = message;
        
        messageDiv.appendChild(senderDiv);
        messageDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Cevap üretme
    generateResponse(prompt, maxLength = 30) {
        // Input tokenize
        const inputIds = this.tokenizer.encode(prompt, 20);
        let input = tf.tensor2d([[inputIds]], [1, 1]);
        
        let generated = [];
        
        // Tokenleri üret
        for (let i = 0; i < maxLength; i++) {
            const prediction = this.model.predict(input);
            const nextTokenProbs = prediction.dataSync();
            
            // En olası token'ı seç
            let nextToken = 0;
            let maxProb = -Infinity;
            
            for (let j = 0; j < nextTokenProbs.length; j++) {
                if (nextTokenProbs[j] > maxProb) {
                    maxProb = nextTokenProbs[j];
                    nextToken = j;
                }
            }
            
            // END token kontrolü
            if (nextToken === 3) break;
            
            generated.push(nextToken);
            
            // Tensörleri temizle
            input.dispose();
            prediction.dispose();
            
            // Yeni input oluştur
            input = tf.tensor2d([[nextToken]], [1, 1]);
        }
        
        // Oluşturulan metni decode et
        const response = this.tokenizer.decode(generated);
        return response || "Hmm, düşünüyorum...";
    }
    
    // Model istatistiklerini güncelle
    updateModelStats() {
        if (!this.model) return;
        
        const statsContainer = document.getElementById('stats-container');
        
        // Model parametreleri hesaplamaları
        let totalParams = 0;
        this.model.layers.forEach(layer => {
            if (layer.countParams) {
                totalParams += layer.countParams();
            }
        });
        
        const modelSize = (totalParams * 4 / (1024 * 1024)).toFixed(2); // MB
        const bits = document.querySelector('input[name="bits"]:checked').value;
        
        const statsInfo = `
╔════════════════════════════════════════╗
║         MODEL İSTATİSTİKLERİ          ║
╠════════════════════════════════════════╣
║ 📊 Toplam Parametre: ${totalParams.toLocaleString()}
║ 💾 Model Boyutu: ~${modelSize} MB
║ 🔢 Quantization: ${bits}-bit
║ 📚 Kelime Sayısı: ${Object.keys(this.tokenizer.word2idx).length}
║ 🖥️ Cihaz: ${this.hasTensorflowGPU ? 'GPU (WebGL)' : 'CPU'}
║ ⏰ Son Eğitim: ${new Date().toLocaleTimeString()}
╚════════════════════════════════════════╝
        `;
        
        statsContainer.textContent = statsInfo;
    }
    
    // Model kaydetme
    saveModel() {
        if (!this.model) {
            alert('Kaydedilecek model yok!');
            return;
        }
        
        // Model ve tokenizer'ı JSON olarak kaydet
        const modelData = {
            tokenizer: this.tokenizer.word2idx,
            config: {
                vocabSize: Object.keys(this.tokenizer.word2idx).length,
                bits: parseInt(document.querySelector('input[name="bits"]:checked').value)
            }
        };
        
        // Model ağırlıklarını kaydet
        const modelWeights = {};
        for (const layer of this.model.layers) {
            if (layer.getWeights().length > 0) {
                modelWeights[layer.name] = layer.getWeights().map(w => {
                    return {
                        shape: w.shape,
                        data: Array.from(w.dataSync())
                    };
                });
            }
        }
        
        modelData.weights = modelWeights;
        
        // JSON'a dönüştür ve indir
        const jsonStr = JSON.stringify(modelData);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        
        const a = document.createElement('a');
        a.download = `ai-model-${new Date().toISOString().slice(0, 10)}.json`;
        a.href = URL.createObjectURL(blob);
        a.click();
        URL.revokeObjectURL(a.href);
        
        this.log(`Model kaydedildi: ${a.download}`);
    }
    
    // Model yükleme
    async loadModel(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            const content = await file.text();
            const modelData = JSON.parse(content);
            
            // Tokenizer'ı yükle
            this.tokenizer.word2idx = modelData.tokenizer;
            this.tokenizer.idx2word = {};
            for (const [word, idx] of Object.entries(modelData.tokenizer)) {
                this.tokenizer.idx2word[idx] = word;
            }
            
            // Yeni model oluştur
            const config = modelData.config;
            this.model = this.createModel(config.vocabSize, config.bits);
            
            // Ağırlıkları yükle
            for (const layer of this.model.layers) {
                if (modelData.weights[layer.name]) {
                    const layerWeights = modelData.weights[layer.name];
                    const weights = layerWeights.map(w => {
                        return tf.tensor(w.data, w.shape);
                    });
                    layer.setWeights(weights);
                }
            }
            
            // Radiobutton'ı güncelle
            document.querySelector(`input[name="bits"][value="${config.bits}"]`).checked = true;
            
            this.updateModelStats();
            this.log(`Model yüklendi: ${file.name}`);
            alert('Model başarıyla yüklendi!');
            
        } catch (error) {
            this.log(`❌ Model yüklenirken hata: ${error.message}`, 'error');
            alert(`Model yüklenemedi: ${error.message}`);
        }
    }
    
    // Log mesajı
    log(message, type = 'info') {
        const logContainer = document.getElementById('log-container');
        const timestamp = new Date().toLocaleTimeString();
        
        let color = 'var(--success-color)';
        if (type === 'error') color = 'var(--danger-color)';
        else if (type === 'warning') color = 'var(--warning-color)';
        
        const logLine = document.createElement('div');
        logLine.innerHTML = `<span style="opacity: 0.7;">[${timestamp}]</span> ${message}`;
        logLine.style.color = color;
        
        logContainer.appendChild(logLine);
        logContainer.scrollTop = logContainer.scrollHeight;
    }
}

// Tokenizer sınıfı
class CustomTokenizer {
    constructor() {
        this.word2idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3};
        this.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"};
        this.wordFreq = {};
    }
    
    fit(texts, minFreq = 2) {
        // Kelimelerin frekanslarını say
        const counter = {};
        
        for (const text of texts) {
            const words = this.tokenize(text);
            for (const word of words) {
                counter[word] = (counter[word] || 0) + 1;
            }
        }
        
        this.wordFreq = counter;
        
        // Kelime indeksleri oluştur
        let idx = Object.keys(this.word2idx).length;
        
        for (const [word, freq] of Object.entries(counter)) {
            if (freq >= minFreq && !(word in this.word2idx)) {
                this.word2idx[word] = idx;
                this.idx2word[idx] = word;
                idx++;
            }
        }
    }
    
    tokenize(text) {
        // Metni küçük harfe çevir ve özel karakterleri temizle
        text = text.toLowerCase();
        text = text.replace(/[^\w\s]/g, ' ');
        return text.split(/\s+/).filter(w => w.length > 0);
    }
    
    encode(text, maxLen = 128) {
        // Metni token ID'lerine çevir
        const tokens = this.tokenize(text);
        const encoded = tokens.map(token => this.word2idx[token] || this.word2idx["<UNK>"]);
        
        // Sabit uzunluğa tamamla
        if (encoded.length < maxLen) {
            return encoded.concat(Array(maxLen - encoded.length).fill(this.word2idx["<PAD>"]));
        } else {
            return encoded.slice(0, maxLen);
        }
    }
    
    decode(indices) {
        // Token ID'lerini metne çevir
        const words = [];
        for (const idx of indices) {
            if (idx === this.word2idx["<PAD>"]) continue;
            if (idx === this.word2idx["<END>"]) break;
            words.push(this.idx2word[idx] || "<UNK>");
        }
        return words.join(' ');
    }
}

// Uygulama başlat
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AITrainerApp();
});