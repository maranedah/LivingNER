import os
from argparse import ArgumentParser
from data import create_data_for_subtask_1, create_data_for_subtask_2, create_data_for_subtask_3

if __name__=='__main__':
    # ToDO: Agregar la opción de utilizar los datos de la carpeta de validación.
    parser = ArgumentParser()
    parser.add_argument('--subtask', type=int, default=1, help='Subtask: 1, 2 or 3')
    parser.add_argument('--output_directory', type=str, default='ner_data', help='Output directory to store the data generated')
    parser.add_argument('--subtask3_entity_type', type=str, default='pet', help='Entity type to generate data')
    
    args = parser.parse_args()
    output_directory = args.output_directory
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    print(f'Creating data for subtask: {args.subtask}')

    if args.subtask == 1:
        create_data_for_subtask_1(output_directory)
    
    if args.subtask == 2:
        create_data_for_subtask_2(output_directory)

    if args.subtask == 3:
        create_data_for_subtask_3(output_directory, args.subtask3_entity_type)

    




    pass