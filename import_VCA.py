from models.Amy_IntermediateRoad import Amy_IntermediateRoad
from simple_utils import load_checkpoint


netVCA = Amy_IntermediateRoad( lowfea_VGGlayer=10, highfea_VGGlayer=36, is_highroad_only=False, is_gist=False)
netVCA = load_checkpoint(netVCA, vca_filepath)
netVCA = netVCA.to(device)


VCA_fake = netVCA(fake).view(-1)