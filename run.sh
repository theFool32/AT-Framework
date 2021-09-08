run_PreAct() {
    python3 -u main.py --model PreActResNet18 --config $1 --project at --gpu $2 &&
    echo "Done"
}

run_WRN() {
    python3 -u main.py --model WideResNet28 --config $1 --project at --gpu $2 &&
    echo "Done"
}

run_Pre_linf() {
    run_PreAct cifar10_linf_AT $1 &&
    run_PreAct cifar10_linf_TRADES $1 &&
    run_PreAct cifar10_linf_AWP_AT $1 &&

    run_PreAct cifar100_linf_AT $1 &&
    run_PreAct cifar100_linf_TRADES $1 &&
    run_PreAct cifar100_linf_AWP_AT $1 &&

    run_PreAct svhn_linf_AT $1 &&
    run_PreAct svhn_linf_TRADES $1 &&
    run_PreAct svhn_linf_AWP_AT $1 &&
    echo "Done"
}
run_Pre_l2() {
    run_PreAct cifar10_l2_AT $1 &&
    run_PreAct cifar10_l2_TRADES $1 &&
    run_PreAct cifar10_l2_AWP_AT $1 &&

    run_PreAct cifar100_l2_AT $1 &&
    run_PreAct cifar100_l2_TRADES $1 &&
    run_PreAct cifar100_l2_AWP_AT $1 &&

    run_PreAct svhn_l2_AT $1 &&
    run_PreAct svhn_l2_TRADES $1 &&
    run_PreAct svhn_l2_AWP_AT $1 &&
    echo "Done"
}
run_WRN_linf() {
    run_WRN cifar10_linf_AT $1 &&
    run_WRN cifar10_linf_TRADES $1 &&
    run_WRN cifar10_linf_AWP_AT $1 &&

    run_WRN cifar100_linf_AT $1 &&
    run_WRN cifar100_linf_TRADES $1 &&
    run_WRN cifar100_linf_AWP_AT $1 &&

    run_WRN svhn_linf_AT $1 &&
    run_WRN svhn_linf_TRADES $1 &&
    run_WRN svhn_linf_AWP_AT $1 &&
    echo "Done"
}
run_WRN_l2() {
    run_WRN cifar10_l2_AT $1 &&
    run_WRN cifar10_l2_TRADES $1 &&
    run_WRN cifar10_l2_AWP_AT $1 &&

    run_WRN cifar100_l2_AT $1 &&
    run_WRN cifar100_l2_TRADES $1 &&
    run_WRN cifar100_l2_AWP_AT $1 &&

    run_WRN svhn_l2_AT $1 &&
    run_WRN svhn_l2_TRADES $1 &&
    run_WRN svhn_l2_AWP_AT $1 &&
    echo "Done"
}


if [ "$1"x == "61"x ]
then
    run_WRN_linf 0 &
    run_WRN_l2 2 &
else
    run_Pre_linf 0 &
    run_Pre_l2 2 &
fi
