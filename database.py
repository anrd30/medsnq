"""
MedQuant Database Module
SQLite schema for storing medical interactions, knowledge base, and benchmarks.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).parent / "medquant.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize the database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Model weights metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_weights (
            model_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            quantization TEXT NOT NULL,
            bits_per_weight REAL,
            file_size_mb REAL,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Patient sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            device_type TEXT DEFAULT 'desktop',
            model_id TEXT,
            FOREIGN KEY (model_id) REFERENCES model_weights(model_id)
        )
    """)
    
    # Interactions log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_query TEXT NOT NULL,
            model_response TEXT NOT NULL,
            symptoms_extracted TEXT,
            inference_time_ms REAL,
            tokens_generated INTEGER,
            tokens_per_second REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES patient_sessions(session_id)
        )
    """)
    
    # Medical conditions table (base content for FTS)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_conditions (
            condition_id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_name TEXT NOT NULL,
            symptoms TEXT,
            description TEXT,
            treatment_guidelines TEXT,
            severity TEXT
        )
    """)
    
    # Full-Text Search virtual table for medical knowledge base
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS medical_kb USING fts5(
            condition_name,
            symptoms,
            description,
            treatment_guidelines,
            content='medical_conditions',
            content_rowid='condition_id'
        )
    """)
    
    # Benchmark results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_results (
            benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT,
            dataset_name TEXT,
            accuracy REAL,
            avg_latency_ms REAL,
            p95_latency_ms REAL,
            tokens_per_second REAL,
            memory_usage_mb REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES model_weights(model_id)
        )
    """)
    
    # Create indexes for efficient querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_time ON interactions(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_model ON benchmark_results(model_id)")
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized at {DB_PATH}")


def seed_medical_conditions():
    """Seed the medical knowledge base with common conditions."""
    conditions = [
        # Respiratory Conditions
        {"condition_name": "Common Cold", "symptoms": "runny nose, sneezing, sore throat, mild fever, cough, congestion", "description": "A viral infection of the upper respiratory tract", "treatment_guidelines": "Rest, fluids, over-the-counter cold medications, typically resolves in 7-10 days", "severity": "low"},
        {"condition_name": "Influenza (Flu)", "symptoms": "high fever, body aches, headache, fatigue, cough, chills, sweating", "description": "A contagious respiratory illness caused by influenza viruses", "treatment_guidelines": "Rest, fluids, antiviral medications if caught early, fever reducers", "severity": "medium"},
        {"condition_name": "Pneumonia", "symptoms": "chest pain, cough with phlegm, fever, shortness of breath, fatigue, confusion", "description": "Infection that inflames air sacs in one or both lungs", "treatment_guidelines": "Antibiotics for bacterial pneumonia, rest, fluids, hospitalization if severe", "severity": "high"},
        {"condition_name": "Bronchitis", "symptoms": "persistent cough, mucus production, chest discomfort, fatigue, shortness of breath", "description": "Inflammation of the lining of bronchial tubes", "treatment_guidelines": "Rest, fluids, cough suppressants, bronchodilators if needed", "severity": "medium"},
        {"condition_name": "Asthma", "symptoms": "wheezing, shortness of breath, chest tightness, coughing, difficulty breathing", "description": "A chronic respiratory condition causing airway inflammation", "treatment_guidelines": "Inhalers (rescue and maintenance), avoiding triggers, action plan for attacks", "severity": "medium"},
        {"condition_name": "Tuberculosis", "symptoms": "persistent cough, coughing blood, chest pain, weight loss, night sweats, fever", "description": "A serious bacterial infection affecting the lungs", "treatment_guidelines": "Long-term antibiotic treatment (6-9 months), isolation during infectious period", "severity": "high"},
        {"condition_name": "Sinusitis", "symptoms": "facial pain, nasal congestion, runny nose, reduced smell, headache, post-nasal drip", "description": "Inflammation of the sinuses often due to infection", "treatment_guidelines": "Nasal decongestants, saline rinse, antibiotics if bacterial, steam inhalation", "severity": "low"},
        {"condition_name": "Chronic Obstructive Pulmonary Disease (COPD)", "symptoms": "chronic cough, shortness of breath, wheezing, chest tightness, frequent respiratory infections", "description": "A group of lung diseases that block airflow", "treatment_guidelines": "Bronchodilators, steroids, pulmonary rehabilitation, oxygen therapy", "severity": "high"},
        {"condition_name": "Allergic Rhinitis", "symptoms": "sneezing, itchy eyes, runny nose, nasal congestion, postnasal drip", "description": "An allergic response causing cold-like symptoms", "treatment_guidelines": "Antihistamines, nasal sprays, avoiding allergens, immunotherapy for severe cases", "severity": "low"},
        {"condition_name": "Laryngitis", "symptoms": "hoarseness, weak voice, voice loss, sore throat, dry cough, tickling sensation", "description": "Inflammation of the voice box from overuse or infection", "treatment_guidelines": "Voice rest, hydration, humidifier, avoid irritants", "severity": "low"},
        
        # Cardiovascular Conditions
        {"condition_name": "Hypertension", "symptoms": "headache, shortness of breath, nosebleeds, dizziness, often asymptomatic", "description": "Persistently elevated blood pressure", "treatment_guidelines": "Lifestyle changes, reduced sodium, exercise, antihypertensive medications", "severity": "high"},
        {"condition_name": "Coronary Artery Disease", "symptoms": "chest pain, angina, shortness of breath, fatigue, heart palpitations", "description": "Narrowing of coronary arteries reducing blood flow to heart", "treatment_guidelines": "Lifestyle changes, medications, angioplasty, bypass surgery if severe", "severity": "high"},
        {"condition_name": "Heart Failure", "symptoms": "shortness of breath, fatigue, swollen legs, rapid heartbeat, persistent cough", "description": "Heart cannot pump blood efficiently to meet body's needs", "treatment_guidelines": "Diuretics, ACE inhibitors, beta-blockers, lifestyle modifications", "severity": "high"},
        {"condition_name": "Arrhythmia", "symptoms": "palpitations, racing heart, slow heartbeat, chest pain, dizziness, fainting", "description": "Irregular heartbeat - too fast, slow, or erratic", "treatment_guidelines": "Medications, cardioversion, ablation, pacemaker if needed", "severity": "medium"},
        {"condition_name": "Peripheral Artery Disease", "symptoms": "leg pain when walking, numbness, coldness in legs, slow healing sores", "description": "Narrowed arteries reducing blood flow to limbs", "treatment_guidelines": "Exercise, medications, angioplasty, lifestyle changes", "severity": "medium"},
        {"condition_name": "Deep Vein Thrombosis", "symptoms": "leg swelling, leg pain, warmth, red or discolored skin, visible veins", "description": "Blood clot in a deep vein, usually in legs", "treatment_guidelines": "Blood thinners, compression stockings, thrombolytics in severe cases", "severity": "high"},
        {"condition_name": "Atrial Fibrillation", "symptoms": "irregular heartbeat, heart palpitations, fatigue, dizziness, shortness of breath", "description": "Irregular and often rapid heart rate", "treatment_guidelines": "Rate control medications, blood thinners, cardioversion, ablation", "severity": "medium"},
        {"condition_name": "Angina Pectoris", "symptoms": "chest pain, chest pressure, pain in arms, neck, jaw, shortness of breath", "description": "Chest pain due to reduced blood flow to heart", "treatment_guidelines": "Nitroglycerin, beta-blockers, lifestyle changes, angioplasty", "severity": "high"},
        {"condition_name": "Hypotension", "symptoms": "dizziness, fainting, blurred vision, nausea, fatigue, lack of concentration", "description": "Abnormally low blood pressure", "treatment_guidelines": "Increase salt and water intake, compression stockings, medications if needed", "severity": "low"},
        {"condition_name": "Varicose Veins", "symptoms": "visible bulging veins, aching legs, heaviness, itching, skin discoloration", "description": "Enlarged, twisted veins usually in legs", "treatment_guidelines": "Compression stockings, elevation, sclerotherapy, surgery if severe", "severity": "low"},
        
        # Gastrointestinal Conditions
        {"condition_name": "Gastroenteritis", "symptoms": "diarrhea, vomiting, stomach cramps, nausea, fever, dehydration", "description": "Inflammation of the stomach and intestines, often called stomach flu", "treatment_guidelines": "Oral rehydration, bland diet, rest, anti-diarrheal medications if needed", "severity": "medium"},
        {"condition_name": "Gastroesophageal Reflux Disease (GERD)", "symptoms": "heartburn, acid reflux, chest pain, difficulty swallowing, regurgitation", "description": "Chronic acid reflux causing stomach acid to flow back", "treatment_guidelines": "Antacids, PPIs, H2 blockers, lifestyle changes, avoid trigger foods", "severity": "medium"},
        {"condition_name": "Peptic Ulcer", "symptoms": "burning stomach pain, bloating, heartburn, nausea, intolerance to fatty foods", "description": "Sores that develop on stomach lining or small intestine", "treatment_guidelines": "PPIs, antibiotics for H. pylori, avoid NSAIDs, dietary changes", "severity": "medium"},
        {"condition_name": "Irritable Bowel Syndrome (IBS)", "symptoms": "abdominal pain, bloating, gas, diarrhea, constipation, mucus in stool", "description": "A common disorder affecting the large intestine", "treatment_guidelines": "Dietary changes, fiber supplements, stress management, medications", "severity": "low"},
        {"condition_name": "Crohn's Disease", "symptoms": "diarrhea, abdominal pain, blood in stool, weight loss, fatigue, fever", "description": "Inflammatory bowel disease affecting digestive tract", "treatment_guidelines": "Anti-inflammatory drugs, immunosuppressants, biologics, surgery if needed", "severity": "high"},
        {"condition_name": "Ulcerative Colitis", "symptoms": "bloody diarrhea, abdominal cramps, rectal pain, urgency, weight loss", "description": "Inflammatory bowel disease affecting colon and rectum", "treatment_guidelines": "Aminosalicylates, corticosteroids, immunomodulators, surgery if severe", "severity": "high"},
        {"condition_name": "Celiac Disease", "symptoms": "diarrhea, bloating, gas, fatigue, weight loss, anemia, skin rash", "description": "Immune reaction to eating gluten", "treatment_guidelines": "Strict gluten-free diet, vitamin supplements, follow-up monitoring", "severity": "medium"},
        {"condition_name": "Appendicitis", "symptoms": "sudden pain around navel moving to lower right, nausea, vomiting, fever, loss of appetite", "description": "Inflammation of the appendix requiring urgent care", "treatment_guidelines": "Emergency surgery (appendectomy), antibiotics", "severity": "high"},
        {"condition_name": "Gallstones", "symptoms": "sudden intense pain in upper right abdomen, back pain, nausea, vomiting", "description": "Hardened deposits in the gallbladder", "treatment_guidelines": "Pain management, surgery to remove gallbladder if symptomatic", "severity": "medium"},
        {"condition_name": "Hemorrhoids", "symptoms": "rectal bleeding, itching, pain, swelling around anus, discomfort", "description": "Swollen blood vessels in rectum or anus", "treatment_guidelines": "High-fiber diet, topical treatments, sitz baths, surgery if severe", "severity": "low"},
        
        # Neurological Conditions
        {"condition_name": "Migraine", "symptoms": "severe headache, nausea, vomiting, light sensitivity, aura, throbbing pain", "description": "A neurological condition causing intense headaches often with other symptoms", "treatment_guidelines": "Pain relievers, anti-nausea medications, rest in dark room, preventive medications", "severity": "medium"},
        {"condition_name": "Tension Headache", "symptoms": "dull aching pain, pressure around forehead, tenderness in scalp, neck pain", "description": "Most common type of headache causing mild to moderate pain", "treatment_guidelines": "OTC pain relievers, stress management, rest, massage, caffeine", "severity": "low"},
        {"condition_name": "Epilepsy", "symptoms": "seizures, temporary confusion, staring spells, uncontrollable jerking, loss of consciousness", "description": "Neurological disorder causing recurrent seizures", "treatment_guidelines": "Anti-seizure medications, surgery, nerve stimulation, ketogenic diet", "severity": "high"},
        {"condition_name": "Parkinson's Disease", "symptoms": "tremor, slowed movement, rigid muscles, impaired posture, speech changes", "description": "Progressive nervous system disorder affecting movement", "treatment_guidelines": "Levodopa, dopamine agonists, physical therapy, deep brain stimulation", "severity": "high"},
        {"condition_name": "Multiple Sclerosis", "symptoms": "numbness, tingling, vision problems, fatigue, difficulty walking, muscle weakness", "description": "Disease where immune system attacks nerve coverings", "treatment_guidelines": "Disease-modifying therapies, steroids for flares, physical therapy", "severity": "high"},
        {"condition_name": "Stroke", "symptoms": "sudden numbness, confusion, trouble speaking, vision problems, severe headache, loss of balance", "description": "Brain damage from interrupted blood supply - medical emergency", "treatment_guidelines": "Emergency treatment, clot-busting drugs, surgery, rehabilitation", "severity": "high"},
        {"condition_name": "Bell's Palsy", "symptoms": "sudden facial weakness, drooping on one side, drooling, eye dryness, taste changes", "description": "Temporary weakness or paralysis of facial muscles", "treatment_guidelines": "Corticosteroids, antiviral medications, eye protection, physical therapy", "severity": "medium"},
        {"condition_name": "Vertigo", "symptoms": "spinning sensation, dizziness, nausea, balance problems, nystagmus, headache", "description": "Sensation of spinning or motion when stationary", "treatment_guidelines": "Epley maneuver, vestibular rehabilitation, medications, treating underlying cause", "severity": "low"},
        {"condition_name": "Carpal Tunnel Syndrome", "symptoms": "numbness in fingers, tingling, weakness, pain in hand and wrist", "description": "Compression of median nerve in wrist", "treatment_guidelines": "Wrist splinting, NSAIDs, corticosteroid injections, surgery if severe", "severity": "low"},
        {"condition_name": "Sciatica", "symptoms": "lower back pain radiating to leg, numbness, tingling, muscle weakness", "description": "Pain along sciatic nerve from lower back through leg", "treatment_guidelines": "Pain medications, physical therapy, hot/cold packs, surgery if severe", "severity": "medium"},
        
        # Endocrine/Metabolic Conditions
        {"condition_name": "Type 2 Diabetes", "symptoms": "increased thirst, frequent urination, fatigue, blurred vision, slow healing", "description": "A metabolic disorder affecting blood sugar regulation", "treatment_guidelines": "Diet management, exercise, oral medications, insulin if needed, regular monitoring", "severity": "high"},
        {"condition_name": "Type 1 Diabetes", "symptoms": "extreme thirst, frequent urination, weight loss, fatigue, blurred vision, mood changes", "description": "Autoimmune condition where pancreas produces little or no insulin", "treatment_guidelines": "Insulin therapy, blood sugar monitoring, carbohydrate counting, regular checkups", "severity": "high"},
        {"condition_name": "Hypothyroidism", "symptoms": "fatigue, weight gain, cold sensitivity, dry skin, constipation, depression, muscle aches", "description": "Underactive thyroid producing insufficient hormones", "treatment_guidelines": "Thyroid hormone replacement (levothyroxine), regular monitoring", "severity": "medium"},
        {"condition_name": "Hyperthyroidism", "symptoms": "weight loss, rapid heartbeat, anxiety, tremor, sweating, heat sensitivity", "description": "Overactive thyroid producing excessive hormones", "treatment_guidelines": "Anti-thyroid medications, radioactive iodine, beta-blockers, surgery", "severity": "medium"},
        {"condition_name": "Cushing's Syndrome", "symptoms": "weight gain in face and trunk, purple stretch marks, high blood pressure, muscle weakness", "description": "Excess cortisol hormone in the body", "treatment_guidelines": "Reduce corticosteroid use, surgery, radiation, medications", "severity": "high"},
        {"condition_name": "Addison's Disease", "symptoms": "fatigue, weight loss, low blood pressure, darkening skin, salt craving, muscle weakness", "description": "Adrenal glands produce insufficient hormones", "treatment_guidelines": "Hormone replacement therapy, increased salt intake during illness", "severity": "high"},
        {"condition_name": "Polycystic Ovary Syndrome (PCOS)", "symptoms": "irregular periods, excess hair growth, acne, weight gain, difficulty getting pregnant", "description": "Hormonal disorder in women of reproductive age", "treatment_guidelines": "Birth control pills, metformin, anti-androgens, lifestyle changes", "severity": "medium"},
        {"condition_name": "Gout", "symptoms": "sudden severe joint pain, swelling, redness, warmth, limited range of motion", "description": "Form of arthritis caused by uric acid crystal deposits", "treatment_guidelines": "NSAIDs, colchicine, corticosteroids, uric acid-lowering medications", "severity": "medium"},
        {"condition_name": "Osteoporosis", "symptoms": "back pain, loss of height, stooped posture, bone fractures, often asymptomatic", "description": "Weakened bones that become fragile and prone to fracture", "treatment_guidelines": "Calcium and vitamin D, bisphosphonates, exercise, fall prevention", "severity": "medium"},
        {"condition_name": "Obesity", "symptoms": "excess body weight, difficulty with physical activity, shortness of breath, increased sweating", "description": "Excessive body fat accumulation affecting health", "treatment_guidelines": "Diet modification, exercise, behavioral therapy, medications, bariatric surgery", "severity": "medium"},
        
        # Infectious Diseases
        {"condition_name": "COVID-19", "symptoms": "fever, cough, fatigue, loss of taste or smell, shortness of breath, body aches", "description": "Respiratory illness caused by SARS-CoV-2 virus", "treatment_guidelines": "Rest, fluids, antivirals if eligible, hospitalization if severe, vaccination for prevention", "severity": "medium"},
        {"condition_name": "Malaria", "symptoms": "high fever, chills, sweating, headache, nausea, vomiting, muscle pain", "description": "Parasitic disease transmitted by mosquitoes", "treatment_guidelines": "Antimalarial medications, supportive care, prevention with prophylaxis", "severity": "high"},
        {"condition_name": "Dengue Fever", "symptoms": "high fever, severe headache, pain behind eyes, joint and muscle pain, rash", "description": "Viral infection transmitted by Aedes mosquitoes", "treatment_guidelines": "Pain relievers (avoid aspirin), fluids, rest, hospital care if severe", "severity": "medium"},
        {"condition_name": "Typhoid Fever", "symptoms": "prolonged fever, weakness, stomach pain, headache, loss of appetite, rash", "description": "Bacterial infection spread through contaminated food or water", "treatment_guidelines": "Antibiotics, fluids, rest, vaccination for prevention", "severity": "high"},
        {"condition_name": "Hepatitis A", "symptoms": "fatigue, nausea, abdominal pain, loss of appetite, jaundice, dark urine", "description": "Liver infection caused by hepatitis A virus", "treatment_guidelines": "Rest, adequate nutrition, avoid alcohol, supportive care", "severity": "medium"},
        {"condition_name": "Hepatitis B", "symptoms": "fatigue, nausea, jaundice, dark urine, abdominal pain, joint pain", "description": "Serious liver infection caused by hepatitis B virus", "treatment_guidelines": "Antiviral medications for chronic cases, monitoring, vaccination for prevention", "severity": "high"},
        {"condition_name": "HIV/AIDS", "symptoms": "flu-like symptoms initially, weight loss, recurring infections, night sweats, fatigue", "description": "Virus that attacks immune system, can lead to AIDS", "treatment_guidelines": "Antiretroviral therapy (ART), regular monitoring, prevention of opportunistic infections", "severity": "high"},
        {"condition_name": "Chickenpox", "symptoms": "itchy rash with blisters, fever, fatigue, headache, loss of appetite", "description": "Highly contagious viral infection causing itchy rash", "treatment_guidelines": "Calamine lotion, antihistamines, antiviral if high risk, vaccination for prevention", "severity": "low"},
        {"condition_name": "Measles", "symptoms": "high fever, cough, runny nose, red eyes, rash spreading from face", "description": "Highly contagious viral disease", "treatment_guidelines": "Supportive care, fever reducers, vitamin A, isolation, vaccination for prevention", "severity": "medium"},
        {"condition_name": "Mumps", "symptoms": "swollen salivary glands, fever, headache, muscle aches, fatigue, loss of appetite", "description": "Viral infection affecting salivary glands", "treatment_guidelines": "Rest, fluids, pain relievers, cold compresses, vaccination for prevention", "severity": "low"},
        
        # Mental Health Conditions
        {"condition_name": "Anxiety Disorder", "symptoms": "excessive worry, restlessness, rapid heartbeat, sweating, difficulty concentrating", "description": "A mental health condition characterized by persistent anxiety", "treatment_guidelines": "Therapy (CBT), anti-anxiety medications, lifestyle changes, relaxation techniques", "severity": "medium"},
        {"condition_name": "Depression", "symptoms": "persistent sadness, loss of interest, fatigue, sleep changes, appetite changes, hopelessness", "description": "A mood disorder causing persistent feelings of sadness", "treatment_guidelines": "Antidepressants, psychotherapy, exercise, social support, lifestyle changes", "severity": "medium"},
        {"condition_name": "Bipolar Disorder", "symptoms": "mood swings, manic episodes, depressive episodes, sleep changes, impulsive behavior", "description": "Mental disorder with extreme mood swings", "treatment_guidelines": "Mood stabilizers, antipsychotics, therapy, lifestyle management", "severity": "high"},
        {"condition_name": "Obsessive-Compulsive Disorder (OCD)", "symptoms": "intrusive thoughts, compulsive behaviors, anxiety, need for order, fear of contamination", "description": "Disorder characterized by obsessions and compulsions", "treatment_guidelines": "CBT with exposure therapy, SSRIs, support groups", "severity": "medium"},
        {"condition_name": "Post-Traumatic Stress Disorder (PTSD)", "symptoms": "flashbacks, nightmares, severe anxiety, avoidance, emotional numbness, hypervigilance", "description": "Mental health condition triggered by traumatic event", "treatment_guidelines": "Trauma-focused therapy, EMDR, medications, support groups", "severity": "high"},
        {"condition_name": "Panic Disorder", "symptoms": "sudden panic attacks, heart palpitations, sweating, trembling, shortness of breath, fear of dying", "description": "Recurrent unexpected panic attacks", "treatment_guidelines": "CBT, anti-anxiety medications, antidepressants, relaxation techniques", "severity": "medium"},
        {"condition_name": "Insomnia", "symptoms": "difficulty falling asleep, waking during night, waking too early, daytime fatigue, irritability", "description": "Sleep disorder characterized by difficulty sleeping", "treatment_guidelines": "Sleep hygiene, CBT for insomnia, short-term sleep aids, treating underlying causes", "severity": "low"},
        {"condition_name": "ADHD", "symptoms": "difficulty focusing, hyperactivity, impulsivity, disorganization, forgetfulness", "description": "Attention deficit hyperactivity disorder affecting focus and behavior", "treatment_guidelines": "Stimulant medications, behavioral therapy, lifestyle modifications, coaching", "severity": "medium"},
        {"condition_name": "Eating Disorders", "symptoms": "extreme food restriction, binge eating, purging, body image distortion, weight changes", "description": "Mental disorders characterized by abnormal eating habits", "treatment_guidelines": "Psychotherapy, nutritional counseling, medications, hospitalization if severe", "severity": "high"},
        {"condition_name": "Social Anxiety Disorder", "symptoms": "intense fear of social situations, avoidance, blushing, sweating, trembling, nausea", "description": "Intense anxiety in social situations", "treatment_guidelines": "CBT, exposure therapy, SSRIs, beta-blockers for performance anxiety", "severity": "medium"},
        
        # Musculoskeletal Conditions
        {"condition_name": "Rheumatoid Arthritis", "symptoms": "joint pain, swelling, stiffness, fatigue, fever, weight loss", "description": "Autoimmune disorder affecting joints", "treatment_guidelines": "DMARDs, biologics, NSAIDs, physical therapy, lifestyle changes", "severity": "high"},
        {"condition_name": "Osteoarthritis", "symptoms": "joint pain, stiffness, tenderness, loss of flexibility, bone spurs, swelling", "description": "Degenerative joint disease from wear and tear", "treatment_guidelines": "Pain relievers, physical therapy, weight management, joint injections, surgery", "severity": "medium"},
        {"condition_name": "Fibromyalgia", "symptoms": "widespread muscle pain, fatigue, sleep problems, memory issues, mood changes", "description": "Chronic condition causing widespread musculoskeletal pain", "treatment_guidelines": "Pain medications, antidepressants, physical therapy, stress management", "severity": "medium"},
        {"condition_name": "Lower Back Pain", "symptoms": "aching pain in lower back, muscle stiffness, limited mobility, pain radiating to legs", "description": "Pain in the lumbar region from various causes", "treatment_guidelines": "Pain relievers, physical therapy, hot/cold therapy, exercise, surgery if severe", "severity": "low"},
        {"condition_name": "Herniated Disc", "symptoms": "back pain, leg pain, numbness, weakness, tingling in affected area", "description": "Spinal disc pushes through outer ring", "treatment_guidelines": "Pain medications, physical therapy, epidural injections, surgery if needed", "severity": "medium"},
        {"condition_name": "Scoliosis", "symptoms": "uneven shoulders, one hip higher, visible curve in spine, back pain, fatigue", "description": "Sideways curvature of the spine", "treatment_guidelines": "Observation, bracing, physical therapy, surgery for severe cases", "severity": "medium"},
        {"condition_name": "Tendinitis", "symptoms": "pain at tendon site, tenderness, mild swelling, stiffness, weakness", "description": "Inflammation or irritation of a tendon", "treatment_guidelines": "Rest, ice, compression, elevation, NSAIDs, physical therapy", "severity": "low"},
        {"condition_name": "Bursitis", "symptoms": "joint pain, swelling, warmth, redness, stiffness, pain with movement", "description": "Inflammation of fluid-filled sacs cushioning joints", "treatment_guidelines": "Rest, ice, pain medications, corticosteroid injections, physical therapy", "severity": "low"},
        {"condition_name": "Plantar Fasciitis", "symptoms": "heel pain, stabbing pain in morning, pain after standing long periods", "description": "Inflammation of tissue connecting heel to toes", "treatment_guidelines": "Stretching exercises, orthotics, night splints, NSAIDs, steroid injections", "severity": "low"},
        {"condition_name": "Rotator Cuff Injury", "symptoms": "shoulder pain, weakness, limited range of motion, pain when lifting", "description": "Injury to muscles and tendons around shoulder joint", "treatment_guidelines": "Rest, physical therapy, NSAIDs, corticosteroid injections, surgery if torn", "severity": "medium"},
        
        # Dermatological Conditions
        {"condition_name": "Eczema", "symptoms": "dry skin, itching, red patches, small raised bumps, thickened skin", "description": "Chronic skin condition causing inflammation", "treatment_guidelines": "Moisturizers, topical corticosteroids, antihistamines, avoiding triggers", "severity": "low"},
        {"condition_name": "Psoriasis", "symptoms": "red patches with silvery scales, dry cracked skin, itching, burning, thickened nails", "description": "Autoimmune condition causing rapid skin cell buildup", "treatment_guidelines": "Topical treatments, light therapy, systemic medications, biologics", "severity": "medium"},
        {"condition_name": "Acne", "symptoms": "blackheads, whiteheads, pimples, cysts, oily skin, scarring", "description": "Skin condition when hair follicles become clogged", "treatment_guidelines": "Benzoyl peroxide, salicylic acid, retinoids, antibiotics, isotretinoin for severe", "severity": "low"},
        {"condition_name": "Rosacea", "symptoms": "facial redness, visible blood vessels, bumps, eye irritation, thickened skin", "description": "Chronic skin condition causing facial redness", "treatment_guidelines": "Topical medications, oral antibiotics, laser therapy, avoiding triggers", "severity": "low"},
        {"condition_name": "Hives (Urticaria)", "symptoms": "raised red welts, intense itching, swelling, welts that change shape", "description": "Skin reaction causing itchy welts", "treatment_guidelines": "Antihistamines, corticosteroids, avoiding triggers, epinephrine for severe", "severity": "low"},
        {"condition_name": "Shingles", "symptoms": "painful rash, blisters, burning sensation, sensitivity to touch, fever, headache", "description": "Viral infection caused by reactivated chickenpox virus", "treatment_guidelines": "Antiviral medications, pain relievers, calamine lotion, cool compresses", "severity": "medium"},
        {"condition_name": "Ringworm", "symptoms": "circular rash, red scaly edges, itching, clear center, multiple rings", "description": "Fungal infection of the skin", "treatment_guidelines": "Antifungal creams, oral antifungals for severe cases, keep area clean and dry", "severity": "low"},
        {"condition_name": "Cellulitis", "symptoms": "red swollen skin, warmth, tenderness, fever, chills, spreading redness", "description": "Bacterial skin infection that can spread", "treatment_guidelines": "Antibiotics, elevation, rest, hospitalization if severe", "severity": "medium"},
        {"condition_name": "Contact Dermatitis", "symptoms": "red rash, itching, dry cracked skin, blisters, swelling, burning", "description": "Skin reaction from contact with irritants or allergens", "treatment_guidelines": "Avoid trigger, topical corticosteroids, moisturizers, antihistamines", "severity": "low"},
        {"condition_name": "Vitiligo", "symptoms": "white patches on skin, premature whitening of hair, loss of color inside mouth", "description": "Condition causing loss of skin color in patches", "treatment_guidelines": "Topical corticosteroids, light therapy, depigmentation, cosmetic cover-ups", "severity": "low"},
        
        # Urological Conditions
        {"condition_name": "Urinary Tract Infection", "symptoms": "burning urination, frequent urination, cloudy urine, pelvic pain, urgency", "description": "Bacterial infection of the urinary system", "treatment_guidelines": "Antibiotics, increased fluid intake, pain relievers, cranberry supplements", "severity": "medium"},
        {"condition_name": "Kidney Stones", "symptoms": "severe pain in side and back, pain during urination, pink or red urine, nausea", "description": "Hard deposits formed in kidneys", "treatment_guidelines": "Pain management, fluids, medications to pass stone, lithotripsy, surgery", "severity": "medium"},
        {"condition_name": "Benign Prostatic Hyperplasia", "symptoms": "frequent urination, difficulty starting urination, weak stream, dribbling, incomplete emptying", "description": "Enlarged prostate gland common in older men", "treatment_guidelines": "Alpha-blockers, 5-alpha reductase inhibitors, minimally invasive therapies, surgery", "severity": "medium"},
        {"condition_name": "Prostatitis", "symptoms": "painful urination, pelvic pain, groin pain, flu-like symptoms, frequent urination", "description": "Inflammation of the prostate gland", "treatment_guidelines": "Antibiotics for bacterial, alpha-blockers, anti-inflammatories, physical therapy", "severity": "medium"},
        {"condition_name": "Overactive Bladder", "symptoms": "sudden urge to urinate, frequent urination, incontinence, nocturia", "description": "Condition causing sudden urge to urinate", "treatment_guidelines": "Bladder training, pelvic floor exercises, medications, botox injections", "severity": "low"},
        {"condition_name": "Chronic Kidney Disease", "symptoms": "fatigue, swelling in feet, poor appetite, nausea, confusion, reduced urine output", "description": "Gradual loss of kidney function over time", "treatment_guidelines": "Managing underlying conditions, diet changes, medications, dialysis, transplant", "severity": "high"},
        {"condition_name": "Interstitial Cystitis", "symptoms": "bladder pain, pelvic pain, frequent urination, urgency, pain during intercourse", "description": "Chronic bladder condition causing pain and pressure", "treatment_guidelines": "Dietary changes, bladder training, medications, physical therapy, bladder instillations", "severity": "medium"},
        {"condition_name": "Erectile Dysfunction", "symptoms": "difficulty getting erection, difficulty maintaining erection, reduced sexual desire", "description": "Inability to achieve or maintain erection", "treatment_guidelines": "PDE5 inhibitors, lifestyle changes, counseling, vacuum devices, surgery", "severity": "low"},
        {"condition_name": "Testicular Torsion", "symptoms": "sudden severe testicular pain, swelling, nausea, vomiting, abdominal pain", "description": "Twisting of spermatic cord cutting blood supply - emergency", "treatment_guidelines": "Emergency surgery within 6 hours, manual detorsion if surgery delayed", "severity": "high"},
        {"condition_name": "Nephrotic Syndrome", "symptoms": "severe swelling around eyes and ankles, foamy urine, weight gain, fatigue", "description": "Kidney disorder causing protein loss in urine", "treatment_guidelines": "Treating underlying cause, ACE inhibitors, diuretics, statins, immunosuppressants", "severity": "high"},
    ]
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if already seeded
    cursor.execute("SELECT COUNT(*) FROM medical_conditions")
    if cursor.fetchone()[0] > 0:
        print("✓ Medical conditions already seeded")
        conn.close()
        return
    
    # Insert conditions
    for condition in conditions:
        cursor.execute("""
            INSERT INTO medical_conditions (condition_name, symptoms, description, treatment_guidelines, severity)
            VALUES (?, ?, ?, ?, ?)
        """, (condition["condition_name"], condition["symptoms"], condition["description"], 
              condition["treatment_guidelines"], condition["severity"]))
    
    # Rebuild FTS index
    cursor.execute("INSERT INTO medical_kb(medical_kb) VALUES('rebuild')")
    
    conn.commit()
    conn.close()
    print(f"✓ Seeded {len(conditions)} medical conditions")


def search_symptoms(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search the medical knowledge base for matching conditions."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT mc.*, bm25(medical_kb) as relevance
        FROM medical_kb
        JOIN medical_conditions mc ON medical_kb.rowid = mc.condition_id
        WHERE medical_kb MATCH ?
        ORDER BY relevance
        LIMIT ?
    """, (query, limit))
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def log_interaction(session_id: str, query: str, response: str, 
                    inference_time_ms: float, tokens: int, model_id: str = None):
    """Log a patient interaction to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Ensure session exists
    cursor.execute("""
        INSERT OR IGNORE INTO patient_sessions (session_id, model_id)
        VALUES (?, ?)
    """, (session_id, model_id))
    
    # Calculate tokens per second
    tps = (tokens / (inference_time_ms / 1000)) if inference_time_ms > 0 else 0
    
    cursor.execute("""
        INSERT INTO interactions (session_id, user_query, model_response, 
                                  inference_time_ms, tokens_generated, tokens_per_second)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, query, response, inference_time_ms, tokens, tps))
    
    conn.commit()
    conn.close()


def get_interaction_stats() -> Dict[str, Any]:
    """Get aggregate statistics about interactions."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_interactions,
            AVG(inference_time_ms) as avg_latency,
            AVG(tokens_per_second) as avg_tps,
            COUNT(DISTINCT session_id) as total_sessions
        FROM interactions
    """)
    
    row = cursor.fetchone()
    stats = dict(row) if row else {}
    conn.close()
    return stats


def register_model(model_id: str, model_name: str, quantization: str, 
                   bits: float, size_mb: float, file_path: str):
    """Register a model in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO model_weights 
        (model_id, model_name, quantization, bits_per_weight, file_size_mb, file_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (model_id, model_name, quantization, bits, size_mb, file_path))
    
    conn.commit()
    conn.close()


def get_registered_models() -> List[Dict[str, Any]]:
    """Get all registered models."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM model_weights ORDER BY file_size_mb")
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


if __name__ == "__main__":
    # Initialize and seed database when run directly
    init_database()
    seed_medical_conditions()
    
    # Test symptom search
    print("\n--- Testing FTS5 Symptom Search ---")
    results = search_symptoms("fever headache")
    for r in results:
        print(f"  • {r['condition_name']} (severity: {r['severity']})")
