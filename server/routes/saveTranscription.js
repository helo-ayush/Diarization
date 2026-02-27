import express from 'express';
import multer from 'multer';
import { cleanAudioForDiarization } from '../utils/audioProcessor.js';
import { transcribeWithDiarization, buildRawTranscript } from '../utils/deepgramProcessor.js';
import { refineTranscript } from '../utils/geminiProcessor.js';

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });


router.post('/', upload.single('call_recording'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).send('No File Uploaded.');
    }

    console.log(`\n${'═'.repeat(60)}`);
    console.log(`📞 Processing audio: ${req.file.originalname} (${(req.file.size / 1024).toFixed(1)} KB)`);
    console.log(`${'═'.repeat(60)}`);

    // STEP 1: Two-path audio processing
    const diarizeBuffer = await cleanAudioForDiarization(req.file.buffer, { forceMono: true })
    console.log(`✅ Audio processed: diarize=${(diarizeBuffer.length / 1024).toFixed(1)}KB`);


    // STEP 2: Deepgram transcription + diarization
    const dgResult = await transcribeWithDiarization(diarizeBuffer);

    if (!dgResult.hasSpeech) {
      return res.json({
        message: "Transcription complete but no speech/speakers detected.",
        transcript: dgResult.transcript
      });
    }

    // STEP 3: Build raw transcript with speaker filtering
    const { rawTranscript, validWords, validSpeakers, segments, lowConfWords } = buildRawTranscript(dgResult.words);

    // sumary =..... 
    // mongo db -> {transcription + summary + embedding}, {transcription + summary + embedding} , {transcription + summary + embedding} 
    // search -> find all customers wo got unsatisfied with our new mmeeting tool -> gemini ->  customer cmpained about meeting , meeting tools failing ...
    // desc -> embedding ....


    // save -> audio (saved in mongo)
    // search -> find all totally satisfied , 50

    console.log('🧠 Sending to Gemini for full technical analysis & refinement...');
    const geminiResult = await refineTranscript(rawTranscript, validSpeakers.length);
    console.log(`✅ Gemini refinement completed in ${geminiResult.processingTime}ms`);

    // Logic for fallbacks
    const finalTranscript = geminiResult.transcript || rawTranscript;
    const wasRefined = geminiResult.transcript !== null;
    const diarizationConfidence = 1 - (lowConfWords / validWords.length);

    // Summary Console Log (Upgraded for visibility)
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`📋 ANALYSIS RESULTS:`);
    console.log(`   Words: ${validWords.length} | Speakers: ${validSpeakers.length}`);
    console.log(`   Satisfaction Score: ${geminiResult.satisfactionScore}/10`);
    console.log(`   Tags: [${geminiResult.tags.join(', ')}]`);
    console.log(`   Gemini refined: ${wasRefined ? '✅ YES' : '❌ NO'}`);
    console.log(`   Confidence: ${(diarizationConfidence * 100).toFixed(1)}%`);
    console.log(`   Summary Length: ${geminiResult.summary?.length || 0} chars`);
    console.log(`${'─'.repeat(60)}\n`);

    res.json({
      success: true,
      // High-signal data for your Vector DB / Frontend
      analysis: {
        summary: geminiResult.summary,
        satisfactionScore: geminiResult.satisfactionScore,
        tags: geminiResult.tags,
      },
      transcript: finalTranscript,
      speakerCount: validSpeakers.length,
      metrics: {
        totalWords: validWords.length,
        rawSegments: segments.length,
        geminiRefined: wasRefined,
        diarizationConfidence: parseFloat(diarizationConfidence.toFixed(3)),
        processingMs: {
          deepgram: dgResult.processingTime,
          gemini: geminiResult.processingTime
        }
      }
    });
  } catch (error) {
    console.error("Transcription Route Error:", error);
    res.status(500).json({ error: "Failed to process audio transcription" });
  }
});

export default router;
