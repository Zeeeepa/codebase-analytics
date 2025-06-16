import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Container,
  Grid,
  TextField,
  Typography,
  Paper,
  Tabs,
  Tab,
  Divider,
  Alert,
} from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { GitHub } from '@mui/icons-material';

// Types
interface RepoData {
  name: string;
  description: string;
  stars: number;
  forks: number;
  issues: number;
  language: string;
  contributors: number;
  commits: number;
  codeQuality: {
    complexity: number;
    maintainability: number;
    testCoverage: number;
    technicalDebt: number;
  };
  languageBreakdown: {
    name: string;
    value: number;
    color: string;
  }[];
  commitActivity: {
    month: string;
    commits: number;
  }[];
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#8DD1E1'];

export default function RepoAnalyticsDashboard() {
  const [repoUrl, setRepoUrl] = useState('');
  const [repoData, setRepoData] = useState<RepoData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setRepoUrl(e.target.value);
  };

  const extractRepoInfo = (url: string) => {
    // Handle different GitHub URL formats
    const githubRegex = /github\.com\/([^\/]+)\/([^\/]+)/;
    const match = url.match(githubRegex);
    
    if (match && match.length >= 3) {
      return {
        owner: match[1],
        repo: match[2].replace('.git', '')
      };
    }
    
    return null;
  };

  const handleFetchRepo = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const repoInfo = extractRepoInfo(repoUrl);
      
      if (!repoInfo) {
        throw new Error('Invalid GitHub repository URL');
      }
      
      // Make API call to backend
      const response = await fetch(`/api/analyze?owner=${repoInfo.owner}&repo=${repoInfo.repo}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setRepoData(data);
    } catch (err) {
      console.error('Error fetching repo data:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          <GitHub sx={{ mr: 1, verticalAlign: 'middle' }} />
          GitHub Repository Analytics
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Enter a GitHub repository URL to analyze code quality, complexity, and development patterns.
        </Typography>
        
        <Box sx={{ display: 'flex', mb: 2 }}>
          <TextField
            fullWidth
            label="GitHub Repository URL"
            variant="outlined"
            value={repoUrl}
            onChange={handleInputChange}
            placeholder="https://github.com/username/repository"
            sx={{ mr: 2 }}
          />
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleFetchRepo}
            disabled={loading || !repoUrl}
          >
            {loading ? <CircularProgress size={24} /> : 'Analyze'}
          </Button>
        </Box>
        
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Paper>

      {repoData && (
        <Box sx={{ mb: 4 }}>
          <Paper elevation={3}>
            <Box sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                {repoData.name}
              </Typography>
              <Typography variant="body1" paragraph>
                {repoData.description}
              </Typography>
              
              <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Stars
                      </Typography>
                      <Typography variant="h5">
                        {repoData.stars}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Forks
                      </Typography>
                      <Typography variant="h5">
                        {repoData.forks}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Open Issues
                      </Typography>
                      <Typography variant="h5">
                        {repoData.issues}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Primary Language
                      </Typography>
                      <Typography variant="h5">
                        {repoData.language}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
              
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={tabValue} onChange={handleTabChange} aria-label="repository analytics tabs">
                  <Tab label="Code Quality" {...a11yProps(0)} />
                  <Tab label="Language Breakdown" {...a11yProps(1)} />
                  <Tab label="Commit Activity" {...a11yProps(2)} />
                </Tabs>
              </Box>
              
              <TabPanel value={tabValue} index={0}>
                <Typography variant="h6" gutterBottom>
                  Code Quality Metrics
                </Typography>
                <Box sx={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { name: 'Complexity', value: repoData.codeQuality.complexity },
                        { name: 'Maintainability', value: repoData.codeQuality.maintainability },
                        { name: 'Test Coverage', value: repoData.codeQuality.testCoverage },
                        { name: 'Technical Debt', value: repoData.codeQuality.technicalDebt },
                      ]}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="value" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </TabPanel>
              
              <TabPanel value={tabValue} index={1}>
                <Typography variant="h6" gutterBottom>
                  Language Distribution
                </Typography>
                <Box sx={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={repoData.languageBreakdown}
                        cx="50%"
                        cy="50%"
                        labelLine={true}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={150}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {repoData.languageBreakdown.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
              </TabPanel>
              
              <TabPanel value={tabValue} index={2}>
                <Typography variant="h6" gutterBottom>
                  Commit Activity (Last 12 Months)
                </Typography>
                <Box sx={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={repoData.commitActivity}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="commits" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </TabPanel>
            </Box>
          </Paper>
        </Box>
      )}
    </Container>
  );
}

