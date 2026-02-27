import 'dotenv/config';
import express from 'express';
import connectDB from './db.js';
import cors from 'cors';
import saveRoute from './routes/saveTranscription.js'

const app = express();

// Middlewares
app.use(cors());
app.use(express.json());


// Connect to MongoDB using Mongoose
connectDB();

// Default Route for health check
app.get('/', (req, res) => res.send("Voice-to-Intent API is running."));

app.use('/save', saveRoute)

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server listening on port ${PORT}`));