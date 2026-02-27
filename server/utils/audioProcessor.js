import ffmpeg from 'fluent-ffmpeg';
import { Readable, Writable } from 'stream';


/**
 * Clean an audio buffer using ffmpeg and return a WAV buffer.
 * Optimized for playback/storage — normalizes volume and removes silence.
 *
 * @param {Buffer} inputBuffer - raw audio data
 * @returns {Promise<Buffer>}
 */
export const cleanAudioBuffer = (inputBuffer) => {
    return new Promise((resolve, reject) => {
        const inputStream = new Readable();
        inputStream.push(inputBuffer);
        inputStream.push(null);

        const chunks = [];
        const outputStream = new Writable({
            write(chunk, encoding, callback) {
                chunks.push(chunk);
                callback();
            }
        });

        ffmpeg(inputStream)
            .toFormat('wav')
            .audioChannels(1)
            .audioFrequency(16000)
            .audioFilters([
                'highpass=f=100',
                'lowpass=f=8000',
                'afftdn=nr=5:nf=-35',
                'dynaudnorm=p=0.9:s=5',
                'compand=attacks=0.3:points=-80/-80|-40/-15|-20/-10|0/-7',
                'silenceremove=start_periods=1:stop_periods=1:start_threshold=-45dB'
            ])
            .on('error', (err) => {
                console.error('FFmpeg error:', err);
                reject(err);
            })
            .on('end', () => {
                resolve(Buffer.concat(chunks));
            })
            .pipe(outputStream);
    });
};


/**
 * Clean audio optimized for DIARIZATION (speaker identification).
 * 
 * KEY DIFFERENCES from cleanAudioBuffer:
 * - NO dynaudnorm/compand: Volume differences between speakers are the #1 signal
 *   the diarizer uses. Equalizing volume destroys this signal.
 * - Stronger noise reduction: Aggressively removes background noise while
 *   preserving vocal characteristics (pitch, timbre).
 * - NO silence removal: Silences between speakers help the diarizer detect
 *   speaker turn boundaries.
 *
 * @param {Buffer} inputBuffer - raw audio data
 * @param {Object} [options] - optional settings
 * @param {boolean} [options.forceMono] - force mono output (default: false)
 * @returns {Promise<Buffer>}
 */
export const cleanAudioForDiarization = (inputBuffer, options = {}) => {
    const { forceMono = false } = options;

    return new Promise((resolve, reject) => {
        const inputStream = new Readable();
        inputStream.push(inputBuffer);
        inputStream.push(null);

        const chunks = [];
        const outputStream = new Writable({
            write(chunk, encoding, callback) {
                chunks.push(chunk);
                callback();
            }
        });

        let cmd = ffmpeg(inputStream)
            .toFormat('wav')
            .audioFrequency(16000);

        if (forceMono) {
            cmd = cmd.audioChannels(1);
        }

        cmd.audioFilters([
            'highpass=f=80',
            'lowpass=f=8000',
            'compand=attacks=0:decays=0.3:points=-80/-80|-40/-40|-35/-10|0/0',
            'afftdn=nr=12:nf=-30:tn=1',
        ])
            .on('error', (err) => {
                console.error('FFmpeg diarization-clean error:', err);
                reject(err);
            })
            .on('end', () => {
                resolve(Buffer.concat(chunks));
            })
            .pipe(outputStream);
    });
};