import { createClient } from "@deepgram/sdk";

const deepgram = createClient(process.env.DEEPGRAM_API_KEY);


// Transcribe audio buffer with diarization using Deepgram Nova-3
export async function transcribeWithDiarization(audioBuffer) {
    const startTime = Date.now();

    console.log(`🎙️ Sending ${(audioBuffer.length / 1024).toFixed(1)}KB to Deepgram...`);

    // Wrap in a timeout so it doesn't hang forever
    const timeoutMs = 60000;
    const deepgramPromise = deepgram.listen.prerecorded.transcribeFile(
        audioBuffer,
        {
            model: "nova-3",
            language: "multi",
            diarize: true,
            smart_format: true,
            punctuate: true,
            utterances: true,
            multichannel: false,
        }
    );

    const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`Deepgram timed out after ${timeoutMs / 1000}s`)), timeoutMs)
    );

    const { result, error } = await Promise.race([deepgramPromise, timeoutPromise]);

    const processingTime = Date.now() - startTime;
    console.log(`✅ Deepgram completed in ${processingTime}ms`);

    if (error) throw error;

    const utterances = result.results?.utterances;
    const channel = result.results?.channels?.[0];
    const wordsRaw = channel?.alternatives?.[0]?.words || [];

    // No speech detected
    if (!utterances || utterances.length === 0) {
        return {
            words: [],
            transcript: channel?.alternatives?.[0]?.transcript || "",
            hasSpeech: false,
            processingTime
        };
    }

    // Build word-level array with speaker info
    const words = wordsRaw.map(w => ({
        word: w.punctuated_word || w.word,
        speaker: w.speaker,
        confidence: w.speaker_confidence ?? w.confidence ?? 0,
        start: w.start,
        end: w.end
    }));

    return { words, hasSpeech: true, processingTime };
}


// Filter out background noise speakers (< 5% word share)
// and build a raw transcript string from valid words
export function buildRawTranscript(words) {
    const totalWords = words.length;

    // Count words per speaker
    const speakerCounts = {};
    words.forEach(w => {
        speakerCounts[w.speaker] = (speakerCounts[w.speaker] || 0) + 1;
    });

    // Filter speakers with < 5% word share
    const validSpeakers = Object.keys(speakerCounts)
        .filter(s => (speakerCounts[s] / totalWords) > 0.05)
        .map(Number);

    console.log(`👥 Speakers detected: ${Object.keys(speakerCounts).length}, valid: ${validSpeakers.length}`);
    for (const [spk, count] of Object.entries(speakerCounts)) {
        const pct = ((count / totalWords) * 100).toFixed(1);
        const valid = validSpeakers.includes(Number(spk));
        console.log(`   Speaker ${spk}: ${count} words (${pct}%) ${valid ? '✅' : '❌ filtered'}`);
    }

    // Filter words and group into speaker segments
    const validWords = words.filter(w => validSpeakers.includes(w.speaker));
    const segments = [];
    let currentSeg = null;

    for (const w of validWords) {
        if (!currentSeg || currentSeg.speaker !== w.speaker) {
            if (currentSeg) segments.push(currentSeg);
            currentSeg = { speaker: w.speaker, words: [w.word] };
        } else {
            currentSeg.words.push(w.word);
        }
    }
    if (currentSeg) segments.push(currentSeg);

    const rawTranscript = segments
        .map(s => `Speaker ${s.speaker}: ${s.words.join(' ')}`)
        .join('\n');

    // Confidence stats
    const avgConfidence = validWords.reduce((a, w) => a + w.confidence, 0) / validWords.length;
    const lowConfWords = validWords.filter(w => w.confidence < 0.7).length;
    console.log(`📊 Deepgram confidence: avg=${avgConfidence.toFixed(3)}, low-conf words=${lowConfWords}/${validWords.length}`);

    return {
        rawTranscript,
        validWords,
        validSpeakers,
        segments,
        avgConfidence,
        lowConfWords
    };
}
