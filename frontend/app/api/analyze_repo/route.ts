import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { repo_url } = body;

    if (!repo_url) {
      return NextResponse.json(
        { error: 'Repository URL is required' },
        { status: 400 }
      );
    }

    // Connect to the backend API
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8666';
    console.log(`Connecting to backend API at ${backendUrl}/analyze`);
    
    // Prepare headers with API key if available
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    
    // Add API key if available
    const apiKey = process.env.API_KEY || 'test_key';
    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }
    
    const backendResponse = await fetch(`${backendUrl}/analyze`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        repo_url,
        analysis_depth: 'comprehensive',
        focus_areas: ['all'],
        include_context: true,
        max_issues: 200,
        enable_ai_insights: true,
      }),
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({}));
      console.error('Backend API error:', errorData);
      return NextResponse.json(
        { 
          error: 'Error from backend API', 
          details: errorData,
          status: backendResponse.status 
        },
        { status: backendResponse.status }
      );
    }

    const data = await backendResponse.json();
    
    // Transform the data to match the frontend's expected format if needed
    // This is where you would map the backend response to the frontend's expected format
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in analyze_repo API route:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: String(error) },
      { status: 500 }
    );
  }
}
