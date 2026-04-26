import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict, Counter
import sys
from statistics import mean, stdev

# Add hospital_council_env to path
sys.path.insert(0, str(Path(__file__).parent))

from hospital_council_env.server.hospital_council_env_environment import HospitalCouncilEnvironment
from hospital_council_env.training.policies import baseline_policy, random_policy
import random as py_random


def run_policy_evaluation(data_root: str, episodes: int, policy_name: str = "baseline"):
    """Run policy evaluation and collect metrics."""
    env = HospitalCouncilEnvironment(data_root=data_root, sample_size=1000)
    
    episode_rewards = []
    episode_steps = []
    scenario_rewards = defaultdict(list)
    scenario_success = Counter()
    action_type_distribution = Counter()
    phase_metrics = defaultdict(lambda: {"rewards": [], "steps": 0})
    
    if policy_name == "baseline":
        policy_fn = baseline_policy
    else:
        rng = py_random.Random(42)
        policy_fn = lambda obs: random_policy(obs, rng)
    
    for episode_idx in range(max(1, episodes)):
        observation = env.reset()
        total_reward = 0.0
        steps = 0
        scenario = observation.scenario_type
        
        while not observation.done:
            action = policy_fn(observation)
            observation = env.step(action)
            total_reward += float(observation.reward or 0.0)
            steps += 1
            
            # Track action types
            if hasattr(action, 'action_type'):
                action_type_distribution[action.action_type] += 1
            
            # Track phase metrics
            phase = getattr(observation, 'phase_name', 'unknown')
            phase_metrics[phase]["rewards"].append(float(observation.reward or 0.0))
            phase_metrics[phase]["steps"] += 1
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        scenario_rewards[scenario].append(total_reward)
        scenario_success[scenario] += (1 if total_reward > 0 else 0)
        
        print(f"Episode {episode_idx + 1}/{episodes}: Reward={round(total_reward, 4)}, Steps={steps}, Scenario={scenario}")
    
    return {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "scenario_rewards": scenario_rewards,
        "scenario_success": scenario_success,
        "action_distribution": dict(action_type_distribution),
        "phase_metrics": phase_metrics,
    }


def create_training_visualizations(metrics: dict, output_dir: str = "artifacts", policy_name: str = "baseline"):
    """Create comprehensive training visualization graphs."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    episode_rewards = metrics["episode_rewards"]
    episode_steps = metrics["episode_steps"]
    scenario_rewards = metrics["scenario_rewards"]
    scenario_success = metrics["scenario_success"]
    action_dist = metrics["action_distribution"]
    phase_metrics = metrics["phase_metrics"]
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Hospital Council Agent Training Visualization ({policy_name} policy)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Episode Rewards Over Time
    ax1 = plt.subplot(3, 3, 1)
    episodes = list(range(1, len(episode_rewards) + 1))
    ax1.plot(episodes, episode_rewards, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    if len(episode_rewards) > 1:
        ax1.axhline(y=mean(episode_rewards), color='r', linestyle='--', label=f'Mean: {mean(episode_rewards):.3f}')
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Total Reward', fontweight='bold')
    ax1.set_title('Episode Rewards Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Episode Length Distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.bar(episodes, episode_steps, color='#A23B72', alpha=0.7)
    ax2.axhline(y=mean(episode_steps), color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean(episode_steps):.1f}')
    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Steps', fontweight='bold')
    ax2.set_title('Episode Length (Steps)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # 3. Scenario Distribution
    ax3 = plt.subplot(3, 3, 3)
    scenario_names = list(scenario_success.keys())
    scenario_counts = [len(scenario_rewards[s]) for s in scenario_names]
    colors = ['#F18F01', '#C73E1D', '#6A994E', '#BC4749']
    ax3.bar(scenario_names, scenario_counts, color=colors, alpha=0.8)
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Scenario Distribution', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    for i, v in enumerate(scenario_counts):
        ax3.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Success Rate by Scenario
    ax4 = plt.subplot(3, 3, 4)
    success_rates = []
    for scenario in scenario_names:
        total = len(scenario_rewards[scenario])
        success = scenario_success[scenario]
        rate = (success / total * 100) if total > 0 else 0
        success_rates.append(rate)
    ax4.barh(scenario_names, success_rates, color='#06A77D', alpha=0.8)
    ax4.set_xlabel('Success Rate (%)', fontweight='bold')
    ax4.set_title('Success Rate by Scenario', fontweight='bold')
    ax4.set_xlim([0, 100])
    for i, v in enumerate(success_rates):
        ax4.text(v + 2, i, f'{v:.1f}%', va='center', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Reward Distribution by Scenario
    ax5 = plt.subplot(3, 3, 5)
    scenario_data = [scenario_rewards[s] for s in scenario_names]
    bp = ax5.boxplot(scenario_data, labels=scenario_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_ylabel('Reward', fontweight='bold')
    ax5.set_title('Reward Distribution by Scenario', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Average Reward by Scenario
    ax6 = plt.subplot(3, 3, 6)
    avg_rewards = [mean(scenario_rewards[s]) if scenario_rewards[s] else 0 for s in scenario_names]
    ax6.bar(scenario_names, avg_rewards, color='#D62828', alpha=0.8)
    ax6.set_ylabel('Average Reward', fontweight='bold')
    ax6.set_title('Average Reward by Scenario', fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    for i, v in enumerate(avg_rewards):
        ax6.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Action Type Distribution
    ax7 = plt.subplot(3, 3, 7)
    if action_dist:
        action_names = list(action_dist.keys())
        action_counts = list(action_dist.values())
        wedges, texts, autotexts = ax7.pie(action_counts, labels=action_names, autopct='%1.1f%%',
                                            colors=['#FFB703', '#FB5607', '#FFBE0B', '#8ECAE6', '#219EBC'],
                                            startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax7.set_title('Action Type Distribution', fontweight='bold')
    
    # 8. Phase Metrics
    ax8 = plt.subplot(3, 3, 8)
    if phase_metrics:
        phases = list(phase_metrics.keys())
        phase_avg_rewards = [mean(phase_metrics[p]["rewards"]) if phase_metrics[p]["rewards"] else 0 for p in phases]
        ax8.barh(phases, phase_avg_rewards, color='#FF006E', alpha=0.8)
        ax8.set_xlabel('Average Reward', fontweight='bold')
        ax8.set_title('Average Reward by Phase', fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_stats = f"""
TRAINING SUMMARY STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Episodes: {len(episode_rewards)}
Policy: {policy_name.upper()}

Rewards:
  Mean: {mean(episode_rewards):.4f}
  Min: {min(episode_rewards):.4f}
  Max: {max(episode_rewards):.4f}
  StdDev: {stdev(episode_rewards) if len(episode_rewards) > 1 else 0:.4f}

Episode Length:
  Mean: {mean(episode_steps):.2f} steps
  Min: {min(episode_steps)} steps
  Max: {max(episode_steps)} steps

Overall Success Rate: {(sum(scenario_success.values()) / len(episode_rewards) * 100):.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """.strip()
    
    ax9.text(0.1, 0.5, summary_stats, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save the figure
    output_path = Path(output_dir) / f"agent_training_visualization_{policy_name}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize Hospital Council agent training")
    parser.add_argument("--data-root", default="physionet.org/files/mimiciv/3.1",
                       help="Path to MIMIC data root")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--policy", default="baseline", choices=["baseline", "random"],
                       help="Policy to evaluate")
    parser.add_argument("--output-dir", default="artifacts",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Hospital Council Agent Training Visualization")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Data Root: {args.data_root}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Policy: {args.policy}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Run evaluation
    print(f"Running {args.policy} policy evaluation...\n")
    metrics = run_policy_evaluation(
        data_root=args.data_root,
        episodes=args.episodes,
        policy_name=args.policy
    )
    
    # Create visualizations
    print(f"\nGenerating training visualizations...")
    create_training_visualizations(metrics, output_dir=args.output_dir, policy_name=args.policy)
    
    print(f"\n{'='*60}")
    print(f"✓ Agent training visualization complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
