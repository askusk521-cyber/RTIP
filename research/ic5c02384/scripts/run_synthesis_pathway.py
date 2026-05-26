import os, sys, shutil
sys.path.insert(0, '/home/lhshen/RTIP/rtipmd/jax/src')

from rtip_jax.system import System
from rtip_jax.config import Para
from rtip_jax.external import DeepMDBoundary, DeepMDPES
from rtip_jax.pes.bias import SynthesisPot
from rtip_jax.workflows import run_rtip_synthesis_path_sampling

IS_PATH = sys.argv[1] if len(sys.argv) > 1 else 'IS.xyz'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('OUTPUT_DIR', '.')
PREFIX = sys.argv[3] if len(sys.argv) > 3 else os.environ.get('OUTPUT_PREFIX', 'synthesis')
MODEL = os.environ.get('MODEL', '/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt')
MAX_STEP = int(os.environ.get('MAX_STEP', '100'))

os.makedirs(OUT_DIR, exist_ok=True)
system = System.read_xyz(IS_PATH)

# 1-tBu has 33 atoms (indices 0-32), CO2 has 3 atoms (indices 33-35)
mol_index = (tuple(range(33)), (33, 34, 35))
print(f'System: {system.natom} atoms, mol_index: {len(mol_index[0])} + {len(mol_index[1])}')

pes = DeepMDPES(DeepMDBoundary(model=MODEL))
e, f = pes.get_energy_force(system)
print(f'Initial energy: {float(e):.6f} Hartree')

config = SynthesisPot(
    initial_state=system,
    mol_index=mol_index,
    para=Para(max_step=MAX_STEP, print_step=1),
    str_output_file=f'{OUT_DIR}/rtip.pdb',
    output_file=f'{OUT_DIR}/rtip.out',
)

print(f'Running synthesis pathway, max_step={MAX_STEP}...')
result = run_rtip_synthesis_path_sampling(config, pes)

print(f'Done. Stopped={result.stopped}, history steps={len(result.history)}')
print(f'Final energy: {float(result.system.pot):.6f} Hartree')

if os.path.exists(f'{OUT_DIR}/rtip.pdb'):
    shutil.move(f'{OUT_DIR}/rtip.pdb', f'{OUT_DIR}/{PREFIX}.pdb')
if os.path.exists(f'{OUT_DIR}/rtip.out'):
    shutil.move(f'{OUT_DIR}/rtip.out', f'{OUT_DIR}/{PREFIX}.out')
print(f'Output: {OUT_DIR}/{PREFIX}.pdb, {OUT_DIR}/{PREFIX}.out')
