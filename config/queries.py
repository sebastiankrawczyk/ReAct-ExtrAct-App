QUERIES = [
  {
    "topic": "Risk of bias Definition: Extract the summary assessment of the study's methodological quality.  Primary Pattern: Look for specific, standardized assessment terms.  Keywords: \"risk of bias\".  Data Format: The output should be one of a limited set of categories, typically \"High\", \"Some concerns\", or \"Low\".",
    "possible_options": ""
  },
  {
    "topic": "Number of participants Definition: Extract the total number of individuals enrolled in the study.  Primary Pattern: Look for the uppercase letter \"N\" followed by an equals sign and a number (e.g., N = 118).  Keywords: \"participants\", \"sample size\", \"enrolled\".  Data Format: The output should be the integer value.",
    "possible_options": ""
  },
  {
    "topic": "Interventions Definition: Extract the names and descriptions of all treatment or control conditions being compared in the study.  Primary Pattern: Identify distinct groups, often labeled with acronyms (e.g., STEPPS, TAU, DBT, DDP) or as a control group (e.g., waitlist).  Keywords: \"intervention\", \"treatment\", \"control group\", \"comparison\".  Data Format: Capture the full description associated with each group, including the frequency, type, and components of the therapy or condition. Each intervention should be extracted as a separate entity.  Example for González-González et al. (2021):  Intervention 1 (STEPPS): \"20 weekly sessions of group STEPPS psychotherapy, five sessions of group psychotherapy for companions, monthly sessions of individual and family psychotherapy, and the possibility of therapy in case of an emergency; this was combined with usual medication and/or psychiatric consultations.\"  Intervention 2 (TAU): \"Individual psychotherapy, medication, and case management.\"",
    "possible_options": ""
  },
  {
    "topic": "Duration Definition: Extract the total length of the intervention period and any specified follow-up periods.  Primary Pattern: Look for a number followed by a unit of time.  Keywords: \"weeks\", \"months\", \"years\", \"duration\", \"follow-up\".  Data Format: Capture the number and the time unit (e.g., \"18 months\", \"20 weeks\", \"12 months\"). If a follow-up is mentioned, extract it as a separate data point (e.g., \"Follow-up: 32 weeks\").",
    "possible_options": ""
  },
  {
    "topic": "Study population Definition: Extract the eligibility and inclusion criteria for the study participants.  Primary Pattern: Look for descriptions of required characteristics.  Keywords: \"diagnosis\", \"criteria\", \"DSM-5\", \"SCID-II\", \"age\", \"males and females\", \"inclusion criteria\".  Data Format: Extract the complete text describing who was eligible for the study, including diagnostic criteria, age ranges, and specific required symptoms or history (e.g., \"self-harm or aggressive impulsive behaviors for the past 2 years\").",
    "possible_options": ""
  },
  {
    "topic": "Setting, country Definition: Extract the location and context where the study was conducted.  Primary Pattern: Look for a geographical location combined with a description of the clinical environment.  Keywords: \"setting\", \"country\", \"outpatient\", \"inpatient\", \"single center\", \"multicenter\", and specific country names (e.g., \"Spain\", \"United States\", \"Canada\").  Data Format: Capture all provided details, such as \"Outpatient, single center, Spain\".",
    "possible_options": "None"
  },
  {
    "topic": "Sample demographics Definition: Extract the descriptive statistics of the participants who were actually enrolled in the study.  Primary Pattern: Look for specific labels followed by a value.  Keywords:  For age: \"Mage\", \"Mean age\"  For gender: \"% female\", \"% male\"  For ethnicity: \"% race/ethnicity\", and specific group names like \"Caucasian\", \"Other\".  Data Format: Extract each demographic as a key-value pair (e.g., Mage: 34, % female: 85). Capture \"NR\" if the data is noted as \"Not Reported\".",
    "possible_options": "None"
  },
  {
    "topic": "Primary outcome Definition: Extract the main variable or measurement used to assess the effect of the interventions.  Primary Pattern: Look for the name of a specific measurement tool, scale, or event, often paired with a time point.  Keywords: \"outcome\", \"primary endpoint\", names of assessment scales (e.g., \"BEST\", \"BSL-23\"), and descriptions of measured events (e.g., \"Frequency of suicidal or nonsuicidal self-injurious episodes\").  Data Format: Capture the full description of the outcome, including the time it was measured (e.g., \"BEST at 12 months\", \"BSL-23 at 20 weeks\").",
    "possible_options": "None"
  }
]
