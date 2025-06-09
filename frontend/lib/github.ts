import { Octokit } from "@octokit/rest"

export interface GitHubRepo {
  id: number
  name: string
  full_name: string
  description: string | null
  private: boolean
  html_url: string
  clone_url: string
  language: string | null
  stargazers_count: number
  forks_count: number
  updated_at: string | null
}

export class GitHubService {
  private octokit: Octokit

  constructor(accessToken: string) {
    this.octokit = new Octokit({
      auth: accessToken,
    })
  }

  async getUserRepos(): Promise<GitHubRepo[]> {
    try {
      const response = await this.octokit.rest.repos.listForAuthenticatedUser({
        sort: 'updated',
        per_page: 100,
        type: 'all'
      })

      return response.data.map(repo => ({
        id: repo.id,
        name: repo.name,
        full_name: repo.full_name,
        description: repo.description,
        private: repo.private,
        html_url: repo.html_url,
        clone_url: repo.clone_url,
        language: repo.language,
        stargazers_count: repo.stargazers_count,
        forks_count: repo.forks_count,
        updated_at: repo.updated_at
      }))
    } catch (error) {
      console.error('Error fetching repositories:', error)
      throw new Error('Failed to fetch repositories')
    }
  }

  async getRepoDetails(owner: string, repo: string) {
    try {
      const response = await this.octokit.rest.repos.get({
        owner,
        repo
      })
      return response.data
    } catch (error) {
      console.error('Error fetching repository details:', error)
      throw new Error('Failed to fetch repository details')
    }
  }
}
