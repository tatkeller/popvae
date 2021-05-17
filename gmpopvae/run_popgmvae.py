import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.gmvae import GMVAE
from codebase.train_pop import train
from codebase.evaluate_pop import evaluate
from pprint import pprint
from torchvision import datasets, transforms
import allel
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--k',         type=int, default=500,   help="Number mixture components in MoG prior")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
parser.add_argument("--infile",
                    help="path to input genotypes in vcf (.vcf | .vcf.gz), \
                          zarr, or .popvae.hdf5 format. Zarr files should be as produced \
                          by scikit-allel's `vcf_to_zarr( )` function. `.popvae.hdf5`\
                          files store filtered genotypes from previous runs (i.e. \
                          from --save_allele_counts).")
parser.add_argument("--max_SNPs",default=None,type=int,
                    help="If not None, randomly select --max_SNPs variants \
                          to run. default: None")
parser.add_argument("--train_prop",default=0.9,type=float,
                    help="proportion of samples to use for training \
                          (vs validation). default: 0.9")
parser.add_argument("--save_allele_counts",default=False,action="store_true",
                    help="save allele counts and and sample IDs to \
                    out+'.popvae.hdf5'.")
parser.add_argument('--meta',     type=str, default=None,     help="meta data for samples")
parser.add_argument('--color_by',     type=str, default="Longitude",     help="color for plot")
parser.add_argument('--group_on',     type=str, default="sampleID",     help="sample column name")
parser.add_argument('--subset',     type=str, default="False",     help="only use a subset of the data")


args = parser.parse_args()
layout = [
    ('model={:s}',  'popgmvae'),
    ('z={:02d}',  args.z),
    ('k={:03d}',  args.k),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)
infile=args.infile
train_prop=args.train_prop
max_SNPs=args.max_SNPs
save_allele_counts=args.save_allele_counts
meta = args.meta
color_by = args.color_by
group_on = args.group_on
subset = args.subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#WIP
if infile.endswith('.zarr'):
    callset = zarr.open_group(infile, mode='r')
    gt = callset['calldata/GT']
    gen = allel.GenotypeArray(gt[:])
    samples = callset['samples'][:]
elif infile.endswith('.vcf') or infile.endswith('.vcf.gz'):
    vcf=allel.read_vcf(infile,log=sys.stderr)
    gen=allel.GenotypeArray(vcf['calldata/GT'])
    samples=vcf['samples']
    del vcf
elif infile.endswith('.popvae.hdf5'):
    h5=h5py.File(infile,'r')
    dc=np.array(h5['derived_counts'])
    samples=np.array(h5['samples'])
    h5.close()

#snp filters
if not infile.endswith('.popvae.hdf5'):
    print("counting alleles")
    ac_all=gen.count_alleles() #count of alleles per snp
    ac=gen.to_allele_counts() #count of alleles per snp per individual

    print("dropping non-biallelic sites")
    biallel=ac_all.is_biallelic()
    dc_all=ac_all[biallel,1] #derived alleles per snp
    dc=np.array(ac[biallel,:,1],dtype="int_") #derived alleles per individual
    missingness=gen[biallel,:,:].is_missing()

    print("dropping singletons")
    ninds=np.array([np.sum(x) for x in ~missingness])
    singletons=np.array([x<=2 for x in dc_all])
    dc_all=dc_all[~singletons]
    dc=dc[~singletons,:]
    ninds=ninds[~singletons]
    missingness=missingness[~singletons,:]
    del singletons

    print("filling missing data with rbinom(2,derived_allele_frequency)")
    # af=np.array([dc_all[x]/(ninds[x]*2) for x in range(dc_all.shape[0])])
    # for i in tqdm.tqdm(range(np.shape(dc)[1])):
    #     indmiss=missingness[:,i]
    #     dc[indmiss,i]=np.random.binomial(2,af[indmiss])

    dc=np.transpose(dc)
    dc=dc*0.5 #0=homozygous reference, 0.5=heterozygous, 1=homozygous alternate

    #save hdf5 for reanalysis
    if save_allele_counts and not infile.endswith('.popvae.hdf5'):
        print("saving derived counts for reanalysis")
        if prune_LD:
            outfile=h5py.File(infile+".LDpruned.popvae.hdf5", "w")
        else:
            outfile=h5py.File(infile+".popvae.hdf5", "w")
        outfile.create_dataset("derived_counts", data=dc)
        outfile.create_dataset("samples", data=samples,dtype=h5py.string_dtype()) #requires h5py >= 2.10.0
        outfile.close()

if not max_SNPs==None:
    print("subsetting to "+str(max_SNPs)+" SNPs")
    dc=dc[:,np.random.choice(range(dc.shape[1]),max_SNPs,replace=False)]

print("running train/test splits")

if subset == "True":
    length = len(dc)
    width = len(dc[0])
    length = int(length)
    width = int(width * 0.1)
    dc = dc[:length, :width]
    samples = samples[:length]

ninds=dc.shape[0]
if train_prop==1:
    train_=np.random.choice(range(ninds),int(train_prop*ninds),replace=False)
    test_=train_
    traingen=dc[train_,:]
    testgen=dc[test_,:]
    trainsamples=samples[train_]
    testsamples=samples[test_]
else:
    train_=np.random.choice(range(ninds),int(train_prop*ninds),replace=False)
    test_=np.array([x for x in range(ninds) if x not in train_])
    traingen=dc[train_,:]
    testgen=dc[test_,:]
    trainsamples=samples[train_]
    testsamples=samples[test_]



# print(train)
# print(test)
# print(traingen) #
# print(testgen) #
# print(trainsamples)
# print(testsamples)

# print(traingen.shape)
# print(testgen.shape)


batch_size = 8

train_loader = torch.utils.data.DataLoader(traingen,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(testgen,
                                          batch_size=1, shuffle=False,
                                          num_workers=0)

full_loader = torch.utils.data.DataLoader(dc,
                                          batch_size=1, shuffle=False,
                                          num_workers=0)

#raise Exception("Stop")

#train_loader, labeled_subset, test_loader = ut.get_mnist_data_and_test(device, use_test_subset=True)
gmvae = GMVAE(nn='popv', encode_dim=len(dc[0]), z_dim=args.z, k=args.k, name=model_name).to(device)

if args.train:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train(model=gmvae,
          train_loader=train_loader,
          labeled_subset=None,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)
    ut.evaluate_lower_bound(gmvae, None, run_iwae=args.train == 2, ox = torch.tensor(testgen))
    evaluate(model=gmvae,
             test_loader=test_loader,
             labeled_subset=None,
             device=device,
             tqdm=tqdm.tqdm,
             iter_max=len(testgen),
             iter_save=args.iter_save,
             samples = testsamples,
             meta = meta,
             group_by = color_by,
             group_on = group_on)

else:
    #writer = ut.prepare_writer(model_name, overwrite_existing=True)
    ut.load_model_by_name(gmvae, global_step=args.iter_max)
    evaluate(model=gmvae,
             test_loader=full_loader,
             labeled_subset=None,
             device=device,
             tqdm=tqdm.tqdm,
             iter_max=len(dc),
             iter_save=args.iter_save,
             samples = samples,
             meta = meta,
             group_by = color_by,
             group_on = group_on)
