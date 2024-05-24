# creates the grappa release and uploads model and datasets to github using github cli.

# throw upon error
set -e

# get an abspath to this script
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

RUN_ID=leif-seute/grappa-1.2/ikj2ps9a


MODELNAME=grappa-1.2.2


# cd to this directory:
pushd $THISDIR

# export model to local models directory
grappa_export --id $RUN_ID --modelname $MODELNAME
grappa_eval --modeltag $MODELNAME
