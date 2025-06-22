

import sys
import argparse
import subprocess

import yaml
import psycopg2

from multirag import *


def parse_args():
    parser = argparse.ArgumentParser(prog='gear-cli', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='multirag commands', dest='stage', required=True)

    embed_parser = subparsers.add_parser(
        'embed',
        description='Embedding Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    embed_parser.set_defaults(func=handle_embedding)
    embed_parser.add_argument(
        '-d',
        '--document-path',
        type=str,
        nargs='?',
        default='articles.json',
        help='Path to the dataset.'
    )
    embed_parser.add_argument(
        '-l',
        '--layers',
        type=int,
        nargs='+',
        default=[31],
        help='Layers to target for the attention heads.'
    )
    embed_parser.add_argument(
        '-o',
        '--output',
        type=str,
        nargs='?',
        default='embeddings.json',
        help='Path to the output file.'
    )
    embed_parser.add_argument(
        '-q',
        '--query-path',
        type=str,
        nargs='?',
        default='queries.json',
        help='Path to the queries.'
    )

    def dist_argtype(s: str) -> DistanceMetric:
        try:
            return DistanceMetric[s.upper()]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"{s!r} is not a valid distance metric.")

    def add_db_args(_parser) -> None:
        _parser.add_argument(
            '-c',
            '--config',
            type=str,
            nargs='?',
            default='config/docker-compose.yaml',
            help='Path to the database Docker compose file.'
        )

        metric_choices = [str(m) for m in DistanceMetric]
        _parser.add_argument(
            '-m',
            '--metric',
            type=dist_argtype,
            nargs='?',
            default=DistanceMetric.COSINE,
            help=f'Distance metric for the vector database, one of {", ".join(metric_choices)}.'
        )

    db_parser = subparsers.add_parser(
        'db',
        description='Database',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    db_parser.set_defaults(func=handle_db)
    add_db_args(db_parser)

    db_subparsers = db_parser.add_subparsers(title='db commands', dest='action', required=True)

    db_subparsers.add_parser(
        'start',
        description='Start database Docker container',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    db_subparsers.add_parser(
        'stop',
        description='Stop database Docker container',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    db_subparsers.add_parser(
        'clear',
        description='Clear database',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    db_import_parser = db_subparsers.add_parser(
        'import',
        description='Import data into the database',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    db_import_parser.add_argument(
        '-e',
        '--embedding-path',
        type=str,
        nargs='?',
        default='embeddings.json',
        help='Path to the embedding data file.'
    )

    eval_parser = subparsers.add_parser(
        'evaluate',
        description='Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    eval_parser.set_defaults(func=handle_evaluation)
    eval_parser.add_argument(
        '-e',
        '--embedding-path',
        type=str,
        nargs='?',
        default='embeddings.json',
        help='Path to the embedding file.'
    )
    eval_parser.add_argument(
        '-l',
        '--layer',
        type=int,
        nargs='?',
        default=31,
        help='Layer to evaluate.'
    )
    eval_parser.add_argument(
        '-o',
        '--output',
        type=str,
        nargs='?',
        default='test-results.json',
        help='Path to the output file.'
    )
    eval_parser.add_argument(
        '-p',
        '--picks',
        type=int,
        nargs='?',
        default=32,
        help='Number of picks.'
    )
    add_db_args(eval_parser)

    return parser.parse_args()



def handle_embedding(args) -> tuple[list[ArticleEmbeddings], list[QueryEmbeddings]]:
    return generate_embeddings(
        article_path=args.document_path,
        query_path=args.query_path,
        target_layers=set(args.layers),
        export_path=args.output
    )


def _initialize_db(args) -> VectorDB:
    with open(args.config, 'r') as file:
        docker_config = yaml.safe_load(file)

    container_config = docker_config['services']['postgres']
    db_config: dict[str, str] = {}
    for param in container_config['environment']:
        key, value = param.split('=')
        db_config[key] = value

    return VectorDB(
        args.metric,
        port=int(container_config['ports'][0].split(':')[0]),
        name=db_config['POSTGRES_DB'],
        user=db_config['POSTGRES_USER'],
        password=db_config['POSTGRES_PASSWORD']
    )


def handle_db(args) -> None:
    if args.action == 'start':
        subprocess.run(["docker-compose", "-f", args.config, "up", "-d"])
        return
    elif args.action == 'stop':
        subprocess.run(["docker-compose", "-f", args.config, "down"])
        return

    try:
        db = _initialize_db(args)
    except psycopg2.OperationalError:
        print("Failed to connect to database. Try 'multirag-cli db start'.", file=sys.stderr)
        return

    if args.action == 'import':
        article_embeddings, query_embeddings = load_embeddings(args.embedding_path)
        db.add_articles(article_embeddings)
    elif args.action == 'clear':
        db.clear()


def handle_evaluation(args) -> dict[str, dict[int, StrategyResult]]:
    db = _initialize_db(args)
    return run_strategies(
        vector_db=db,
        embedding_path=args.embedding_path,
        layer=args.layer,
        num_picks=args.picks,
        export_path=args.output
    )


def handle_plotting(args) -> None:
    plot_all(
        data_path=args.data_path,
        export_dir=args.output,
        file_format=args.format
    )


def main():
    args = parse_args()

    if args.func is None:
        raise NotImplementedError(f'Stage {args.stage} not implemented')

    args.func(args)


if __name__ == '__main__':
    main()
