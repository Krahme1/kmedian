args = {}
def get_input_arguments():
    
    args["problem_family"] = get_problem_family()
    args["algorithm"] = get_algorithm()
    args["parameters"] = get_parameters(args["algorithm"])
    args["dataset"] = get_datasets()
    args["use_gpu"] = get_gpu_setting()

    return args


def get_problem_family():
    print("\nSelect problem family:")
    print("\t1: K-Median Problem (KMP)")
    print("\t2: K-Center Problem (KCP)")
    print("\t3: Capacitated K-Median Problem (CKMP)")
    print("\t4: K-Facility Location Problem (KFLP)")

    return input("Enter a value 1-4: ")


def get_algorithm():
    print("Select Algorithm:")
    if args["problem_family"] == "1":
        print("\t1: Modified Hopfield Algorithm")
        print("\t2: Haralampiev's Algorithm")
        print("\t3: Local Search Algorithm")
        print("\t4: Zhu's algorithm MRA")
        print("\t5: Arya Multi Swap")
        print("\t6: Cohen-Addad Local Search")
        print("\t7: Cohen-Addad's Multi Swap")
        print("\t8: 2nk Original Single Hopfield")
        print("\t9: 2nk Best Half Single Hopfield")
        print("\t10: 2nk Best Half Multi Hopfield")
        print("\t11: 2nk Best Half Second Closest Hopfield")
        print("\t12: 2nk Exhaustive Hopfield")
        print("\t13: Fast Interchange")
        print("\t14: Dominguez NAL")
        print("\t15: Parallel Hopfield Algorithm")
        print("\t16: PAM Solver")
        print("\t17: Parallel Hopfield Second Parallel Solver")
        print("\t18: Parallel Hopfield Third Parallel Solver")
        return input("Enter a value 1-18: ")
    
    elif args["problem_family"] == "2":
        print("\t1: 2nk Original Single Hopfield K-center")
        print("\t2: Local Search Algorithm K-center")
        print("\t3: Parallel Hopfield Algorithm K-center")
        print("\t4: Farthest-First K-Center")
        print("\t5: Farthest Client Reassignment K-Center")
        print("\t6: Greedy Add Remove K-Center")
        print("\t7: Drop Worst Facility K-Center")
        print("\t8: Local Recenter K-Center")
        print("\t9: Randomized Swap K-Center")
        return input("Enter a value 1-9: ")
    
    elif args["problem_family"] == "3":
        print("\t1: 2nk Original Single Hopfield Capacitated K-median")
        print("\t2: Parallel Hopfield Capacitated K-median")
        return input("Enter a value 1-2: ")
    
    elif args["problem_family"] == "4":
        print("\t1: Parallel Hopfield K-Facility Location")
        print("\t2: Samei Solis Oba Multi Swap K-Facility Location")
        print("\t3: Zhang's Multi Swap K-Facility Location")
        return input("Enter a value 1-3: ")
    
    else:
        print("Invalid problem family selection.")
        return None

def get_parameters(algorithm_selection):
    parameters = {}

    if args["problem_family"] == "1":
        # n + n^2 Hopfield parameters
        if algorithm_selection == "1":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        # Haralampiev Parameters
        elif algorithm_selection == "2":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            input_temp = input("\nEnter Starting Temperature (Default is 1.0): ")
            if len(input_temp) == 0:
                print("Using default: 1.0")
                parameters["temperature"] = 1.0
            else:
                parameters["temperature"] = float(input_temp)
            input_epoch = input("\nEnter Epoch Length integer. Leave empty to use the dataset n values: ")
            if len(input_epoch) == 0:
                print("Using default: n")
                parameters["epoch"] = None
            else:
                parameters["epoch"] = int(input_epoch)
            input_decay = input("\nEnter Decay Interval (Default is 3): ")
            if len(input_decay) == 0:
                print("Using default: 3")
                parameters["decay"] = 3
            else:
                parameters["decay"] = int(input_decay)

            return parameters
        
        # Local Search Parameters
        elif algorithm_selection == "3":
            max_time = input("\nEnter Maximum runtime in seconds (Default is 5): ")
            if len(max_time) == 0:
                print("Using default: 5")
                parameters["max_time"] = 5
            else:
                parameters["max_time"] = int(max_time)
            return parameters
        
        # Zhu's algorith MRA
        elif algorithm_selection == "4":
            max_time = input("\nEnter Maximum runtime in seconds (Default is 5): ")
            if len(max_time) == 0:
                print("Using default: 5")
                parameters["max_time"] = 5
            else:
                parameters["max_time"] = int(max_time)
            return parameters
            
        # Arya Multi Swap
        elif algorithm_selection == "5":
            max_time = input("\nEnter Maximum runtime in seconds (Default is 5): ")
            if len(max_time) == 0:
                print("Using default: 5")
                parameters["max_time"] = 5
            else:
                parameters["max_time"] = int(max_time)
            return parameters
            
        # Cohen-Addad Local Search
        elif algorithm_selection == "6":
            max_time = input("\nEnter Maximum runtime in seconds (Default is 5): ")
            if len(max_time) == 0:
                print("Using default: 5")
                parameters["max_time"] = 5
            else:
                parameters["max_time"] = int(max_time)
            return parameters
            
        # Cohen-Addad Multi Swap
        elif algorithm_selection == "7":
            max_time = input("\nEnter Maximum runtime in seconds (Default is 5): ")
            if len(max_time) == 0:
                print("Using default: 5")
                parameters["max_time"] = 5
            else:
                parameters["max_time"] = int(max_time)
            return parameters
            
        # 2nk Original Single Hopfield parameters
        if algorithm_selection == "8":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
            
        # 2nk Best Half Single Hopfield parameters
        if algorithm_selection == "9":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
            
        # 2nk Best Half Multi Hopfield parameters
        if algorithm_selection == "10":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
            
        # 2nk Best Half Single Second Closest Hopfield parameters
        if algorithm_selection == "11":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
            
        # 2nk Exhaustive Hopfield parameters
        if algorithm_selection == "12":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
            
        # Fast Interchange
        if algorithm_selection == "13":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
            
        # Dominguez NAL
        if algorithm_selection == "14":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters

        if algorithm_selection == "15":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "16":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "17":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "18":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
    elif args["problem_family"] == "2":
        if algorithm_selection == "1":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "2":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "3":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "4":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "5":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "6":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "7":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "8":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        if algorithm_selection == "9":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
    elif args["problem_family"] == "3":
        if algorithm_selection == "1":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        if algorithm_selection == "2":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
    elif args["problem_family"] == "4":
        if algorithm_selection == "1":
            input_runs = input("\nEnter Number of Runs (Default is 1): ")
            if len(input_runs) == 0:
                print("Using default: 1 run")
                parameters["runs"] = 1
            else:
                parameters["runs"] = int(input_runs)
            return parameters
        
        elif algorithm_selection == "2":
            swap_size = input("\nEnter Swap Size for Multi Swap (Default is 1): ")
            if len(swap_size) == 0:
                print("Using default: 1")
                parameters["swap_size"] = 1
            else:
                parameters["swap_size"] = int(swap_size)
            return parameters
        
        elif algorithm_selection == "3":
            swap_size = input("\nEnter Swap Size for Multi Swap (Default is 1): ")
            if len(swap_size) == 0:
                print("Using default: 1")
                parameters["swap_size"] = 1
            else:
                parameters["swap_size"] = int(swap_size)

            epsilon_prime = input("\nEnter Epsilon Prime for Zhang's Algorithm (Default is 0.1): ")
            if len(epsilon_prime) == 0:
                print("Using default: 0.1")
                parameters["epsilon_prime"] = 0.1
            else:                
                parameters["epsilon_prime"] = float(epsilon_prime)
            return parameters


    return None


def get_datasets():
    print("\nSelect Dataset:")
    print("\t1: Random-Small")
    print("\t2: Random-Large")
    print("\t3: USCA312")
    print("\t4: OR-Library P-Median")
    print("\t5: TSPLib")
    print("\t6: Special")

    return input("Enter a value 1-6: ")


def get_gpu_setting():
    print("\nUse the GPU (requires CUDA)?")
    selection = input("Enter a value y/n: ")

    if selection.lower() == "y":
        return True
    else:
        return False
