-- Table for storing project submissions
CREATE TABLE IF NOT EXISTS submissions (
    id SERIAL PRIMARY KEY,
    project_name VARCHAR(255) NOT NULL,
    file_hierarchy JSONB NOT NULL,  -- Complete file tree structure
    submission_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    clip_vectorstore_path TEXT,
    CONSTRAINT unique_project_submission UNIQUE (project_name, submission_timestamp)
);

-- Table for storing evaluations
CREATE TABLE IF NOT EXISTS evaluations (
    id SERIAL PRIMARY KEY,
    project_name VARCHAR(255) NOT NULL,
    file_list JSONB NOT NULL,  -- Store file paths as JSONB array
    criteria_matches JSONB NOT NULL,  -- Store criteria matches as JSONB object
    summary TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_project_timestamp UNIQUE (project_name, timestamp)
);

-- Create indices for better query performance
CREATE INDEX IF NOT EXISTS idx_submissions_project_name 
ON submissions(project_name);

CREATE INDEX IF NOT EXISTS idx_submissions_timestamp 
ON submissions(submission_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_evaluations_project_name 
ON evaluations(project_name);

CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp 
ON evaluations(timestamp DESC);

-- Add GIN indices for JSONB fields to improve JSON search performance
CREATE INDEX IF NOT EXISTS idx_submissions_file_hierarchy
ON submissions USING GIN (file_hierarchy);

CREATE INDEX IF NOT EXISTS idx_evaluations_file_list 
ON evaluations USING GIN (file_list);

CREATE INDEX IF NOT EXISTS idx_evaluations_criteria_matches 
ON evaluations USING GIN (criteria_matches);