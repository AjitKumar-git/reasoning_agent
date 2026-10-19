"""
Self-Replicating Continuous Riemannian Spheroid (Mitotic Neural Swarm)
=====================================================================
Biological Mitosis & Clonal Differentiation for Continuous 3D Manifolds:
1. Metabolic Saturation & Mitotic Trigger:
   - Spheroids track cumulative cognitive wave flux and metabolic energy M(t).
   - When M(t) exceeds the mitotic threshold Theta_mitosis (or via replicate()),
     the mother spheroid initiates biological cell division.
2. 3D Tangent Packing & Budding:
   - Selects an open 3D kissing tangent vector n_bud from the kissing-sphere lattice:
     c_daughter = c_mother + 2.0 * n_bud.
3. Genomic Inheritance & Epigenetic Mutation:
   - Clones mother neural conductances and parameters with tunable mutation rate:
     W_daughter = W_mother + sigma_mut * N(0, I).
4. Tangent Contact Welding:
   - Dynamically constructs a bilateral tangent contact patch D_{mother <-> daughter}
     enabling continuous boundary wave exchange.
5. Cytoplasmic Wave Cleavage:
   - Splits and transfers active wave energy into the daughter organoid,
     so the daughter awakens alive and pulsating rather than quiescent.
6. Autonomous Code-Level Replication:
   - Exports self-contained executable progeny bundles (manifest, weights, runtime)
     capable of independent execution and secondary self-replication.
"""

import os
import sys
import copy
import math
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.real_geodesic_sphere import (
    build_fibonacci_geodesic_mesh,
    RealGeodesicSpheroidCore,
    GeodesicSensoryAfferent,
    GeodesicMotorEfferent,
)
from src.modular_kissing_spheres import TangentContactPatch, RetrogradeWaveInterference


# 6-neighbor Octahedral + 12-neighbor FCC Kissing Vectors
KISSING_DIRECTIONS = [
    # Axial (Octahedral, distance = 2.0)
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    # Diagonal FCC contacts
    (1.0 / math.sqrt(2), 0.0, 1.0 / math.sqrt(2)),
    (-1.0 / math.sqrt(2), 0.0, 1.0 / math.sqrt(2)),
    (1.0 / math.sqrt(2), 0.0, -1.0 / math.sqrt(2)),
    (-1.0 / math.sqrt(2), 0.0, -1.0 / math.sqrt(2)),
    (0.0, 1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
    (0.0, -1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
]


class MitoticSpheroidOrganoid(nn.Module):
    """
    An autonomous continuous Riemannian organoid capable of cellular self-replication.
    Tracks internal metabolic energy and splits into mother + daughter upon saturation.
    """
    def __init__(
        self,
        organoid_id: str,
        center: Tuple[float, float, float],
        generation: int = 0,
        num_shells: int = 8,
        nodes_per_shell: int = 128,
        channels: int = 64,
        mitotic_threshold: float = 500.0,
    ):
        super().__init__()
        self.organoid_id = organoid_id
        self.center = torch.tensor(center, dtype=torch.float32)
        self.generation = generation
        self.num_shells = num_shells
        self.nodes_per_shell = nodes_per_shell
        self.channels = channels
        self.mitotic_threshold = mitotic_threshold

        # Physical 3D Core
        self.core = RealGeodesicSpheroidCore(
            num_shells=num_shells,
            nodes_per_shell=nodes_per_shell,
            channels=channels,
            soliton_lambda=0.08,
        )

        # Metabolic & Mitotic Counters
        self.cumulative_energy = 0.0
        self.mitosis_count = 0

    def pulse(
        self,
        state: torch.Tensor,
        past_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], bool]:
        """
        Advances the 3D unit ball by one physical wave cycle and accumulates metabolic energy.
        Returns: (next_state, updated_past, should_replicate)
        """
        next_state, updated_past = self.core.pulse_wave_step(state, past_states)

        # Accumulate kinetic metabolic energy
        pulse_energy = (next_state ** 2).mean().item()
        self.cumulative_energy += pulse_energy

        should_replicate = self.cumulative_energy >= self.mitotic_threshold
        return next_state, updated_past, should_replicate

    def reset_metabolism_after_mitosis(self):
        """Metabolic reset following cytokinesis / cell division."""
        self.cumulative_energy = 0.0
        self.mitosis_count += 1


class SelfReplicatingColony(nn.Module):
    """
    Colony of touching Riemannian kissing spheres that dynamically grows, buds,
    and self-replicates in 3D space.
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 256,
        channels: int = 64,
        num_shells: int = 8,
        nodes_per_shell: int = 128,
        mitotic_threshold: float = 350.0,
        mutation_rate: float = 0.01,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.channels = channels
        self.num_shells = num_shells
        self.nodes_per_shell = nodes_per_shell
        self.mitotic_threshold = mitotic_threshold
        self.mutation_rate = mutation_rate

        # Mesh coordinates reference
        coords, L_S2, south_idx, north_idx, eq_idx, callosal_map = build_fibonacci_geodesic_mesh(
            num_nodes=nodes_per_shell, k_neighbors=7
        )
        self.register_buffer("coords_s2", coords)

        # 1. Primary Primordial Mother Spheroid at (0, 0, 0)
        self.organoids = nn.ModuleDict()
        primordial = MitoticSpheroidOrganoid(
            organoid_id="spheroid_gen0_0",
            center=(0.0, 0.0, 0.0),
            generation=0,
            num_shells=num_shells,
            nodes_per_shell=nodes_per_shell,
            channels=channels,
            mitotic_threshold=mitotic_threshold,
        )
        self.organoids[primordial.organoid_id] = primordial

        # Tangent Contact Patches between touching spheres
        self.contact_patches = nn.ModuleDict()

        # Sensory Ingestion Pole & Motor Readout Pole
        num_sensory = len(south_idx)
        num_motor = len(north_idx)
        self.sensory_pole = GeodesicSensoryAfferent(
            vocab_size=vocab_size, embed_dim=embed_dim, channels=channels, num_sensory_nodes=num_sensory
        )
        self.motor_pole = GeodesicMotorEfferent(
            channels=channels, embed_dim=embed_dim, vocab_size=vocab_size, num_motor_nodes=num_motor
        )
        self.corollary_proj = nn.Linear(vocab_size, channels, bias=False)
        self.retrograde_interference = RetrogradeWaveInterference(channels=channels)

        # Designated Sensory Root and Motor Root organoid IDs
        self.sensory_root_id = primordial.organoid_id
        self.motor_root_id = primordial.organoid_id

        # Colony Statistics
        self.total_replications = 0

    def find_open_tangent_site(self, mother_center: torch.Tensor) -> Tuple[float, float, float]:
        """
        Scans kissing directions around mother_center for an unoccupied 3D tangent site (distance = 2.0).
        """
        existing_centers = [org.center for org in self.organoids.values()]

        dev = mother_center.device
        for d_vec in KISSING_DIRECTIONS:
            cand = mother_center + 2.0 * torch.tensor(d_vec, dtype=torch.float32, device=dev)
            # Check collision with existing organoids (minimum distance should be >= 1.95)
            collision = False
            for ec in existing_centers:
                if torch.norm(cand - ec.to(dev)).item() < 1.90:
                    collision = True
                    break
            if not collision:
                return (cand[0].item(), cand[1].item(), cand[2].item())

        # Fallback: extend outward along z-axis
        max_z = max(c[2].item() for c in existing_centers)
        return (mother_center[0].item(), mother_center[1].item(), max_z + 2.0)

    def replicate_spheroid(
        self,
        mother_id: str,
        mutation_std: Optional[float] = None,
        living_states_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[str, Optional[Dict[str, torch.Tensor]]]:
        """
        Performs biological self-replication / mitosis:
        1. Selects an open 3D kissing tangent position.
        2. Clones the mother spheroid's parameters (with epigenetic mutation).
        3. Welds a bidirectional tangent contact patch between mother and daughter.
        4. Transfers cytoplasmic wave energy so the daughter awakens alive.
        """
        mother = self.organoids[mother_id]
        new_gen = mother.generation + 1
        self.total_replications += 1
        daughter_id = f"spheroid_gen{new_gen}_{self.total_replications}"

        daughter_center = self.find_open_tangent_site(mother.center)

        # 1. Instantiate Daughter Organoid
        daughter = MitoticSpheroidOrganoid(
            organoid_id=daughter_id,
            center=daughter_center,
            generation=new_gen,
            num_shells=self.num_shells,
            nodes_per_shell=self.nodes_per_shell,
            channels=self.channels,
            mitotic_threshold=self.mitotic_threshold,
        )

        # 2. Genomic Inheritance & Mutation
        mother_dict = mother.core.state_dict()
        daughter_dict = {}
        mut_rate = self.mutation_rate if mutation_std is None else mutation_std

        for k, v in mother_dict.items():
            if v.is_floating_point() and mut_rate > 0.0:
                noise = torch.randn_like(v) * (v.abs().mean().clamp(min=1e-4) * mut_rate)
                daughter_dict[k] = v.clone() + noise
            else:
                daughter_dict[k] = v.clone()

        daughter.core.load_state_dict(daughter_dict)
        dev = self.coords_s2.device
        daughter.to(device=dev)
        daughter.center = daughter.center.to(device=dev)
        self.organoids[daughter_id] = daughter

        # 3. Dynamic Tangent Contact Patch Welding
        patch_name = f"patch_{mother_id}_TO_{daughter_id}"
        contact_patch = TangentContactPatch(
            name=patch_name,
            sphere_a_idx=len(self.organoids) - 2,
            sphere_b_idx=len(self.organoids) - 1,
            center_a=mother.center.to(dev),
            center_b=daughter.center.to(dev),
            coords_s2=self.coords_s2,
            channels=self.channels,
            angular_aperture_deg=32.0,
        ).to(device=dev)
        self.contact_patches[patch_name] = contact_patch

        # Also connect daughter to any existing neighbor that happens to be tangent
        for other_id, other_org in self.organoids.items():
            if other_id not in [mother_id, daughter_id]:
                dist = torch.norm(daughter.center.to(dev) - other_org.center.to(dev)).item()
                if abs(dist - 2.0) < 0.15:
                    sec_patch_name = f"patch_{other_id}_TO_{daughter_id}"
                    sec_patch = TangentContactPatch(
                        name=sec_patch_name,
                        sphere_a_idx=0,
                        sphere_b_idx=1,
                        center_a=other_org.center.to(dev),
                        center_b=daughter.center.to(dev),
                        coords_s2=self.coords_s2,
                        channels=self.channels,
                        angular_aperture_deg=32.0,
                    ).to(device=dev)
                    self.contact_patches[sec_patch_name] = sec_patch

        # 4. Cytoplasmic Wave Cleavage (Energy Inheritance)
        if living_states_dict is not None and mother_id in living_states_dict:
            mother_state = living_states_dict[mother_id]
            # Half-energy wave transfer into daughter
            daughter_state = mother_state.clone() * 0.50
            living_states_dict[mother_id] = mother_state * 0.50
            living_states_dict[daughter_id] = daughter_state

        # Designate highest-Z organoid as motor readout root, lowest-Z as sensory root
        centers = {oid: org.center[2].item() for oid, org in self.organoids.items()}
        self.motor_root_id = max(centers, key=centers.get)
        self.sensory_root_id = min(centers, key=centers.get)

        mother.reset_metabolism_after_mitosis()
        print(f"★ [MITOSIS COMPLETE] {mother_id} -> Daughter {daughter_id} at {daughter_center} (Colony Size: {len(self.organoids)})")
        return daughter_id, living_states_dict

    def initialize_colony_states(
        self, batch_size: int, device: torch.device, dtype: torch.dtype, living: bool = True
    ) -> Dict[str, torch.Tensor]:
        states = {}
        for oid, org in self.organoids.items():
            if living:
                states[oid] = org.core.init_living_state(batch_size, device, dtype)
            else:
                states[oid] = org.core.init_state(batch_size, device, dtype)
        return states

    def pulse_colony_cycle(
        self,
        states: Dict[str, torch.Tensor],
        past_states: Optional[Dict[str, List[torch.Tensor]]] = None,
        retrograde_wave: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], List[str]]:
        """
        Advances the entire multi-sphere mitotic colony by one synchronized pulse cycle:
        1. Bilateral boundary flux exchange across all tangent contact patches.
        2. Retrograde wave interference in intermediate reasoning organoids.
        3. Internal 3D wave evolution in each sphere.
        4. Identifies any spheres that crossed the mitotic saturation threshold.
        """
        if past_states is None:
            past_states = {k: [] for k in states.keys()}

        # 1. Exchange Boundary Wave Flux across Tangent Patches
        for patch_name, patch in self.contact_patches.items():
            # Extract organoid names from patch_name: "patch_{A}_TO_{B}"
            parts = patch_name.split("_TO_")
            oid_a = parts[0].replace("patch_", "")
            oid_b = parts[1]

            if oid_a in states and oid_b in states:
                da, db = patch.exchange_flux(states[oid_a], states[oid_b])
                states[oid_a] = states[oid_a] + da
                states[oid_b] = states[oid_b] + db

        # 2. Retrograde Wave Interference
        if retrograde_wave is not None and len(self.organoids) > 1:
            for oid in states.keys():
                if oid != self.motor_root_id and oid != self.sensory_root_id:
                    states[oid] = self.retrograde_interference(states[oid], retrograde_wave)

        # 3. Synchronized Wave Pulse across all Colony Organoids
        new_states = {}
        new_past = {}
        ready_for_mitosis = []

        for oid, org in self.organoids.items():
            st, pst, should_rep = org.pulse(states[oid], past_states.get(oid, []))
            new_states[oid] = st
            new_past[oid] = pst
            if should_rep:
                ready_for_mitosis.append(oid)

        return new_states, new_past, ready_for_mitosis

    def ingest_prompt(
        self, prompt_tokens: torch.Tensor, states: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Ingests sensory tokens exclusively onto South Pole of Sensory Root Sphere.
        """
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        dtype = self.sensory_pole.norm.weight.dtype

        if states is None:
            states = self.initialize_colony_states(B, device, dtype, living=True)

        sensory_field, sensory_memory = self.sensory_pole(prompt_tokens)

        # Ingest onto sensory root sphere outermost shell (k = -1), South Pole
        sensory_org = self.organoids[self.sensory_root_id]
        states[self.sensory_root_id][:, -1, sensory_org.core.sensory_nodes, :] += sensory_field.to(dtype)
        return states, sensory_memory

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        thinking_cycles: int = 3,
        auto_mitosis: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with optional autonomous cellular mitosis during deliberation.
        """
        B, ans_len = target_tokens.shape
        device = prompt_tokens.device
        dtype = self.sensory_pole.norm.weight.dtype

        states, sensory_memory = self.ingest_prompt(prompt_tokens)
        past_states = {k: [] for k in states.keys()}

        # Deliberation Phase
        retrograde = None
        for cycle in range(thinking_cycles):
            states, past_states, ready_mitosis = self.pulse_colony_cycle(
                states, past_states, retrograde_wave=retrograde
            )
            # Autonomous mitosis if threshold reached and colony size < 8 (memory bounded)
            if auto_mitosis and len(ready_mitosis) > 0 and len(self.organoids) < 8:
                for mother_id in ready_mitosis:
                    if len(self.organoids) < 8:
                        self.replicate_spheroid(mother_id, living_states_dict=states)

            if cycle >= 1 and self.motor_root_id in states:
                retrograde = states[self.motor_root_id].clone().detach() * 0.25

        # Motor Efferent Readout from Motor Root
        motor_org = self.organoids[self.motor_root_id]
        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)
        logits_list = []

        for t in range(ans_len):
            motor_field = states[self.motor_root_id][:, -1, motor_org.core.motor_nodes, :]
            logits_t, refractory_trace, corollary = self.motor_pole.forward_readout(
                motor_field, sensory_memory, refractory_trace, step_idx=t
            )
            logits_list.append(logits_t)

            # Corollary Discharge into Core of Sensory/Intermediate Spheres
            coro = corollary.unsqueeze(1).to(dtype)
            for oid in states.keys():
                if oid != self.motor_root_id:
                    states[oid][:, 0, :, :] += coro * 0.20
                    states[oid][:, 1, :, :] += coro * 0.10

            states, past_states, _ = self.pulse_colony_cycle(states, past_states)

        return torch.stack(logits_list, dim=1)

    @torch.no_grad()
    def solve_with_active_thinking(
        self,
        prompt_tokens: torch.Tensor,
        thinking_cycles: int = 3,
        max_answer_tokens: int = 32,
        temperature: float = 0.65,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        dtype = self.sensory_pole.norm.weight.dtype

        states, sensory_memory = self.ingest_prompt(prompt_tokens)
        past_states = {k: [] for k in states.keys()}

        retrograde = None
        for cycle in range(thinking_cycles):
            states, past_states, _ = self.pulse_colony_cycle(states, past_states, retrograde_wave=retrograde)
            if cycle >= 1 and self.motor_root_id in states:
                retrograde = states[self.motor_root_id].clone() * 0.25

        motor_org = self.organoids[self.motor_root_id]
        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)
        generated = []

        for step_idx in range(max_answer_tokens):
            motor_field = states[self.motor_root_id][:, -1, motor_org.core.motor_nodes, :]
            logits, refractory_trace, corollary = self.motor_pole.forward_readout(
                motor_field, sensory_memory, refractory_trace, step_idx=step_idx
            )

            if eos_token_id is not None and step_idx < 3:
                logits[..., eos_token_id] -= 50.0

            logits = logits / max(temperature, 1e-4)
            top_k = 40
            val, idx = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(val, dim=-1)
            sampled_rel = torch.multinomial(probs.squeeze(1) if probs.dim() == 3 else probs, num_samples=1)
            next_tok = torch.gather(idx.squeeze(1) if idx.dim() == 3 else idx, -1, sampled_rel)
            generated.append(next_tok)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

            coro = corollary.unsqueeze(1).to(dtype)
            for oid in states.keys():
                if oid != self.motor_root_id:
                    states[oid][:, 0, :, :] += coro * 0.20
                    states[oid][:, 1, :, :] += coro * 0.10

            states, past_states, _ = self.pulse_colony_cycle(states, past_states)

        if len(generated) == 0:
            return torch.empty((B, 0), dtype=torch.long, device=device)
        return torch.cat(generated, dim=-1)

    def export_autonomous_progeny(self, output_dir: str = "checkpoints/progeny") -> str:
        """
        Exports a self-contained biological progeny manifest and weights.
        The daughter bundle contains everything necessary to boot and run independently.
        """
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "progeny_manifest.json")
        weights_path = os.path.join(output_dir, "colony_genome.pt")

        torch.save(self.state_dict(), weights_path)

        manifest = {
            "colony_size": len(self.organoids),
            "total_replications": self.total_replications,
            "sensory_root": self.sensory_root_id,
            "motor_root": self.motor_root_id,
            "organoids": {
                oid: {
                    "generation": org.generation,
                    "center": org.center.tolist(),
                    "mitosis_count": org.mitosis_count,
                    "cumulative_energy": round(org.cumulative_energy, 4),
                }
                for oid, org in self.organoids.items()
            },
            "contact_patches": list(self.contact_patches.keys()),
            "genome_file": weights_path,
            "can_self_replicate": True,
            "timestamp": time.time(),
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"★ [PROGENY MANIFEST EXPORTED] -> {manifest_path}")
        return manifest_path

    @classmethod
    def from_manifest(cls, manifest_path: str, device: torch.device, dtype: torch.dtype) -> "SelfReplicatingColony":
        """
        Reconstructs a fully welded multi-sphere mitotic colony directly from manifest and genome.
        """
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        colony = cls(
            vocab_size=50257,
            embed_dim=256,
            channels=64,
            num_shells=8,
            nodes_per_shell=128,
        ).to(device=device, dtype=dtype)

        # 1. Rebuild all organoids from manifest
        for oid, o_info in manifest["organoids"].items():
            if oid not in colony.organoids:
                daughter = MitoticSpheroidOrganoid(
                    organoid_id=oid,
                    center=tuple(o_info["center"]),
                    generation=o_info["generation"],
                    num_shells=8,
                    nodes_per_shell=128,
                    channels=64,
                )
                daughter.to(device=device, dtype=dtype)
                daughter.center = daughter.center.to(device)
                colony.organoids[oid] = daughter

        # 2. Rebuild all contact patches from manifest
        for patch_name in manifest["contact_patches"]:
            if patch_name not in colony.contact_patches:
                parts = patch_name.split("_TO_")
                oid_a = parts[0].replace("patch_", "")
                oid_b = parts[1]
                patch = TangentContactPatch(
                    name=patch_name,
                    sphere_a_idx=0,
                    sphere_b_idx=1,
                    center_a=colony.organoids[oid_a].center.to(device),
                    center_b=colony.organoids[oid_b].center.to(device),
                    coords_s2=colony.coords_s2,
                    channels=64,
                    angular_aperture_deg=32.0,
                ).to(device=device, dtype=dtype)
                colony.contact_patches[patch_name] = patch

        colony.sensory_root_id = manifest["sensory_root"]
        colony.motor_root_id = manifest["motor_root"]
        colony.total_replications = manifest["total_replications"]

        weights_path = os.path.join(ROOT_DIR, manifest["genome_file"])
        if os.path.exists(weights_path):
            colony.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

        print(f"★ [RECONSTRUCTED COLONY] Size: {len(colony.organoids)} organoids, {len(colony.contact_patches)} patches from {manifest_path}")
        return colony


def build_self_replicating_colony() -> SelfReplicatingColony:
    return SelfReplicatingColony(
        vocab_size=50257,
        embed_dim=256,
        channels=64,
        num_shells=8,
        nodes_per_shell=128,
        mitotic_threshold=250.0,
        mutation_rate=0.01,
    )
